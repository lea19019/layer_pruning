#!/usr/bin/env python3
"""Autonomous experiment orchestrator powered by Claude Code.

Manages the full lifecycle of all experiments: submit SLURM jobs, poll for
completion, invoke Claude Code to validate results / diagnose and fix failures,
then continue to the next experiment.

Run inside tmux so it survives SSH disconnects:
    tmux new -s experiments
    cd ~/attention_lp && source .venv/bin/activate
    python scripts/orchestrator.py
    # Ctrl+B, D to detach

Reconnect anytime:
    tmux attach -t experiments
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_DIR / "experiments" / "results" / "orchestrator_state.json"
LOG_FILE = PROJECT_DIR / "logs" / "orchestrator.log"
CONFIGS_DIR = PROJECT_DIR / "experiments" / "configs"
RESULTS_DIR = PROJECT_DIR / "experiments" / "results"

POLL_INTERVAL = 120          # seconds between squeue checks
MAX_CONCURRENT_JOBS = 3      # be a good cluster citizen
MAX_RETRIES = 3              # per experiment
CLAUDE_MAX_TURNS = 30        # limit Claude's tool-use loops

# SLURM scripts per experiment type
SLURM_SCRIPTS = {
    "heuristic": "scripts/slurm/run_heuristic.sh",
    "default": "scripts/slurm/run_experiment.sh",
}

# Experiment dependency graph: experiment -> list of prerequisite experiment IDs
# "data" and "ifr_scores" and "kd_data" are virtual prerequisites.
PREREQS = {
    "data": [],
    "ifr_scores": ["data"],
    "kd_data": ["data"],
    "B0": ["data"], "B1": ["data"],
}

# Build M/I/L prereqs programmatically
for n in [8, 12, 16]:
    for v in [1, 3]:
        PREREQS[f"M{v}_{n}"] = ["data"]
        PREREQS[f"I{v}_{n}"] = ["ifr_scores"]
    for v in [2, 4]:
        PREREQS[f"M{v}_{n}"] = ["data", "kd_data"]
        PREREQS[f"I{v}_{n}"] = ["ifr_scores", "kd_data"]
    for v in [1, 2, 3, 4]:
        PREREQS[f"L{v}_{n}"] = ["ifr_scores"]  # placeholder

PREREQS["I5_threshold"] = ["ifr_scores"]

# Map virtual prereqs to their SLURM scripts
VIRTUAL_JOBS = {
    "data": "scripts/slurm/data_prep.sh",
    "ifr_scores": "scripts/slurm/ifr_score.sh",
    "kd_data": "scripts/slurm/generate_kd.sh",
}

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orchestrator")


# ── State Management ─────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "session_id": None,
        "experiments": {},
        "virtual_jobs": {},
        "started_at": datetime.now().isoformat(),
    }


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_exp_state(state: dict, exp_id: str) -> dict:
    if exp_id not in state["experiments"]:
        state["experiments"][exp_id] = {
            "status": "pending",    # pending | submitted | running | completed | failed | skipped
            "slurm_job_id": None,
            "retries": 0,
            "last_error": None,
            "validated": False,
        }
    return state["experiments"][exp_id]


# ── SLURM Helpers ────────────────────────────────────────────────────────────

def sbatch(script: str, args: list[str] = None, partition: str = None) -> str | None:
    """Submit a SLURM job, return job ID or None on failure."""
    cmd = ["sbatch", "--parsable"]
    if partition:
        cmd += [f"--partition={partition}", f"--qos={'cs' if partition == 'cs' else 'gpu'}"]
        if partition == "cs":
            # Override GPU type for cs partition
            cmd += ["--gres=gpu:a100:1"]
    cmd.append(str(PROJECT_DIR / script))
    if args:
        cmd.extend(args)

    log.info(f"  Submitting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_DIR)

    if result.returncode != 0:
        log.error(f"  sbatch failed: {result.stderr.strip()}")
        return None

    job_id = result.stdout.strip().split(";")[0]  # parsable format
    log.info(f"  Submitted job {job_id}")
    return job_id


def get_job_states() -> dict[str, str]:
    """Query squeue for all our jobs. Returns {job_id: state}."""
    result = subprocess.run(
        ["squeue", "-u", subprocess.getoutput("whoami"), "--format=%i %T", "--noheader"],
        capture_output=True, text=True,
    )
    jobs = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            jobs[parts[0]] = parts[1]
    return jobs


def get_job_exit_code(job_id: str) -> int | None:
    """Get exit code of a completed SLURM job."""
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=ExitCode", "--noheader", "--parsable2", "-n"],
        capture_output=True, text=True,
    )
    for line in result.stdout.strip().splitlines():
        code = line.split(":")[0]
        try:
            return int(code)
        except ValueError:
            continue
    return None


# ── Claude Code Integration ──────────────────────────────────────────────────

def call_claude(prompt: str, session_id: str | None = None) -> tuple[str, str]:
    """Call Claude Code in headless mode. Returns (response_text, session_id)."""
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--max-turns", str(CLAUDE_MAX_TURNS),
        "--allowedTools",
        "Read,Edit,Write,Glob,Grep,"
        "Bash(python *),Bash(python3 *),"
        "Bash(source *),Bash(cat *),Bash(head *),Bash(tail *),Bash(wc *),Bash(ls *),"
        "Bash(sbatch *),Bash(squeue *),Bash(sacct *),Bash(srun *),Bash(sinfo *),Bash(scancel *),"
        "Bash(git status*),Bash(git log*),Bash(git diff*),Bash(git add *),Bash(git commit *),"
        "Bash(mkdir *),Bash(nvidia-smi*)",
    ]
    if session_id:
        cmd += ["--resume", session_id]

    log.info(f"  Calling Claude Code (session={session_id or 'new'})...")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=PROJECT_DIR,
        timeout=600,  # 10 min max per Claude call
    )

    output = result.stdout.strip()

    # Parse JSON response
    try:
        # Find the last JSON object in output (there may be noise before it)
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("{") and '"session_id"' in line:
                data = json.loads(line)
                return data.get("result", ""), data.get("session_id", session_id)
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: return raw output
    return output, session_id


def claude_validate_results(exp_id: str, session_id: str | None) -> tuple[bool, str, str]:
    """Ask Claude to validate experiment results. Returns (is_valid, reason, session_id)."""
    results_path = RESULTS_DIR / exp_id / "results.json"
    log_pattern = f"logs/experiment_*.out"

    prompt = f"""Experiment {exp_id} has completed. Validate the results:

1. Read {results_path} and check:
   - COMET score is between 0.0 and 1.0 (and > 0.1 for non-trivial models)
   - chrF++ is between 0 and 100
   - BLEU is between 0 and 100
   - Sample translations look like actual German text (not hallucinated repetitions,
     not source language echoed back, not meta-commentary)
   - num_layers makes sense for the experiment config

2. If this is an experiment with pruning, fine-tuning, or quantization, the metrics
   should be in a reasonable range compared to what you'd expect. Catastrophic drops
   (e.g. COMET < 0.1) suggest something went wrong.

3. Check sample_translations: each hypothesis should be a single clean German translation.
   If you see repeated text, "Czech:" appearing in hypotheses, or translations that are
   3x+ longer than references, flag it.

Respond with EXACTLY one of these formats:
VALID: <brief reason>
INVALID: <what's wrong and what likely caused it>"""

    response, session_id = call_claude(prompt, session_id)

    is_valid = response.strip().upper().startswith("VALID")
    return is_valid, response.strip(), session_id


def claude_diagnose_and_fix(exp_id: str, job_id: str, session_id: str | None) -> tuple[bool, str, str]:
    """Ask Claude to diagnose a failed experiment and fix the code if possible.
    Returns (was_fixed, summary, session_id)."""
    err_file = list(Path(PROJECT_DIR / "logs").glob(f"*_{job_id}.err"))
    out_file = list(Path(PROJECT_DIR / "logs").glob(f"*_{job_id}.out"))

    err_path = err_file[0] if err_file else "not found"
    out_path = out_file[0] if out_file else "not found"

    prompt = f"""Experiment {exp_id} FAILED (SLURM job {job_id}).

1. Read the error log at {err_path} and the output log at {out_path}
   (read the last 100 lines of each if they're long).

2. Diagnose the root cause. Common issues:
   - OOM (out of memory) → may need smaller batch size or different partition
   - Missing files/dependencies
   - Code bugs (IndexError, KeyError, etc.)
   - CUDA errors
   - Timeout

3. If it's a code bug, FIX IT directly by editing the source files. Then run
   `python -m pytest tests/ -x -q` to make sure your fix doesn't break tests.

4. If it's an environment/resource issue (OOM, timeout), explain what needs to change
   but don't modify SLURM scripts (the orchestrator handles retry logic).

Respond with EXACTLY one of these formats:
FIXED: <what you changed and why>
NEEDS_RETRY: <reason, e.g. OOM - will retry with different settings>
UNFIXABLE: <why this can't be auto-fixed>"""

    response, session_id = call_claude(prompt, session_id)

    was_fixed = response.strip().upper().startswith("FIXED") or \
                response.strip().upper().startswith("NEEDS_RETRY")
    return was_fixed, response.strip(), session_id


# ── Orchestration Logic ──────────────────────────────────────────────────────

def get_slurm_script(exp_id: str) -> str:
    """Determine which SLURM script to use for an experiment."""
    if exp_id in VIRTUAL_JOBS:
        return VIRTUAL_JOBS[exp_id]
    if exp_id.startswith("M"):
        return SLURM_SCRIPTS["heuristic"]
    return SLURM_SCRIPTS["default"]


def get_config_path(exp_id: str) -> str | None:
    """Get the YAML config path for an experiment."""
    if exp_id in VIRTUAL_JOBS:
        return None  # virtual jobs don't take config args
    path = CONFIGS_DIR / f"{exp_id}.yaml"
    return str(path) if path.exists() else None


def prereqs_met(exp_id: str, state: dict) -> bool:
    """Check if all prerequisites for an experiment are completed."""
    prereqs = PREREQS.get(exp_id, [])
    for prereq in prereqs:
        if prereq in VIRTUAL_JOBS:
            vj = state.get("virtual_jobs", {}).get(prereq, {})
            if vj.get("status") != "completed":
                return False
        else:
            es = state["experiments"].get(prereq, {})
            if es.get("status") != "completed":
                return False
    return True


def count_active_jobs(state: dict) -> int:
    """Count experiments currently submitted or running."""
    count = 0
    for es in state["experiments"].values():
        if es["status"] in ("submitted", "running"):
            count += 1
    for vj in state.get("virtual_jobs", {}).values():
        if vj.get("status") in ("submitted", "running"):
            count += 1
    return count


def has_results(exp_id: str) -> bool:
    """Check if an experiment already has results."""
    return (RESULTS_DIR / exp_id / "results.json").exists()


def discover_experiments() -> list[str]:
    """Get ordered list of all experiment IDs to run."""
    # Virtual jobs first, then configs in dependency order
    virtual = ["data", "ifr_scores", "kd_data"]
    configs = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
    return virtual + configs


def run_orchestrator():
    """Main orchestration loop."""
    state = load_state()
    all_experiments = discover_experiments()

    log.info("=" * 60)
    log.info("EXPERIMENT ORCHESTRATOR STARTED")
    log.info(f"Total experiments: {len(all_experiments)}")
    log.info(f"State file: {STATE_FILE}")
    log.info(f"Poll interval: {POLL_INTERVAL}s")
    log.info("=" * 60)

    # Check for already-completed experiments (e.g. from previous runs)
    for exp_id in all_experiments:
        if exp_id in VIRTUAL_JOBS:
            continue
        es = get_exp_state(state, exp_id)
        if es["status"] == "pending" and has_results(exp_id):
            log.info(f"  {exp_id}: results already exist, marking completed")
            es["status"] = "completed"

    # Check if data/ifr_scores/kd_data already exist
    if (PROJECT_DIR / "data" / "filtered" / "test.cs").exists():
        state.setdefault("virtual_jobs", {})["data"] = {"status": "completed", "slurm_job_id": None}
        log.info("  data: already exists")
    if (RESULTS_DIR / "ifr_scores.json").exists():
        state.setdefault("virtual_jobs", {})["ifr_scores"] = {"status": "completed", "slurm_job_id": None}
        log.info("  ifr_scores: already exists")
    if (PROJECT_DIR / "data" / "kd").exists() and any((PROJECT_DIR / "data" / "kd").iterdir()):
        state.setdefault("virtual_jobs", {})["kd_data"] = {"status": "completed", "slurm_job_id": None}
        log.info("  kd_data: already exists")

    save_state(state)

    while True:
        # ── 1. Poll SLURM for job status updates ─────────────────────────
        active_jobs = get_job_states()
        jobs_changed = False

        # Check virtual jobs
        for vj_id, vj_state in state.get("virtual_jobs", {}).items():
            if vj_state.get("status") not in ("submitted", "running"):
                continue
            job_id = vj_state.get("slurm_job_id")
            if not job_id:
                continue

            if job_id in active_jobs:
                new_status = "running" if active_jobs[job_id] == "RUNNING" else "submitted"
                if vj_state["status"] != new_status:
                    vj_state["status"] = new_status
                    jobs_changed = True
            else:
                # Job no longer in queue → completed or failed
                exit_code = get_job_exit_code(job_id)
                if exit_code == 0:
                    log.info(f"✓ {vj_id} (job {job_id}) COMPLETED")
                    vj_state["status"] = "completed"
                else:
                    log.error(f"✗ {vj_id} (job {job_id}) FAILED (exit={exit_code})")
                    vj_state["status"] = "failed"
                jobs_changed = True

        # Check experiment jobs
        for exp_id in all_experiments:
            if exp_id in VIRTUAL_JOBS:
                continue
            es = get_exp_state(state, exp_id)
            if es["status"] not in ("submitted", "running"):
                continue

            job_id = es["slurm_job_id"]
            if not job_id:
                continue

            if job_id in active_jobs:
                new_status = "running" if active_jobs[job_id] == "RUNNING" else "submitted"
                if es["status"] != new_status:
                    log.info(f"  {exp_id} (job {job_id}): {es['status']} → {new_status}")
                    es["status"] = new_status
                    jobs_changed = True
            else:
                # Job gone from queue → completed or failed
                exit_code = get_job_exit_code(job_id)

                if exit_code == 0 and has_results(exp_id):
                    log.info(f"✓ {exp_id} (job {job_id}) COMPLETED — validating results...")

                    is_valid, reason, state["session_id"] = claude_validate_results(
                        exp_id, state.get("session_id")
                    )

                    if is_valid:
                        log.info(f"  {exp_id}: {reason}")
                        es["status"] = "completed"
                        es["validated"] = True
                    else:
                        log.warning(f"  {exp_id} INVALID RESULTS: {reason}")
                        es["status"] = "failed"
                        es["last_error"] = f"Validation failed: {reason}"
                else:
                    log.error(f"✗ {exp_id} (job {job_id}) FAILED (exit={exit_code})")
                    es["status"] = "failed"

                    # Ask Claude to diagnose and fix
                    was_fixed, summary, state["session_id"] = claude_diagnose_and_fix(
                        exp_id, job_id, state.get("session_id")
                    )
                    log.info(f"  Claude diagnosis: {summary[:200]}")
                    es["last_error"] = summary

                jobs_changed = True

        if jobs_changed:
            save_state(state)

        # ── 2. Submit new jobs if slots available ────────────────────────
        n_active = count_active_jobs(state)

        # Submit virtual jobs first
        for vj_id in ["data", "ifr_scores", "kd_data"]:
            if n_active >= MAX_CONCURRENT_JOBS:
                break
            vj = state.setdefault("virtual_jobs", {}).setdefault(vj_id, {"status": "pending"})
            if vj["status"] != "pending":
                continue
            # Check prereqs
            vj_prereqs = PREREQS.get(vj_id, [])
            prereqs_ok = all(
                state.get("virtual_jobs", {}).get(p, {}).get("status") == "completed"
                for p in vj_prereqs
            )
            if not prereqs_ok:
                continue

            script = VIRTUAL_JOBS[vj_id]
            job_id = sbatch(script)
            if job_id:
                vj["status"] = "submitted"
                vj["slurm_job_id"] = job_id
                n_active += 1
                log.info(f"▶ Submitted {vj_id} → job {job_id}")
                save_state(state)

        # Submit experiment jobs
        for exp_id in all_experiments:
            if exp_id in VIRTUAL_JOBS:
                continue
            if n_active >= MAX_CONCURRENT_JOBS:
                break

            es = get_exp_state(state, exp_id)

            # Skip completed or actively running
            if es["status"] in ("completed", "submitted", "running", "skipped"):
                continue

            # Retry failed experiments (up to MAX_RETRIES)
            if es["status"] == "failed":
                if es["retries"] >= MAX_RETRIES:
                    if es["status"] != "skipped":
                        log.warning(f"  {exp_id}: max retries ({MAX_RETRIES}) reached, skipping")
                        es["status"] = "skipped"
                        save_state(state)
                    continue
                # Only retry if Claude said it was fixable
                if es.get("last_error", "").upper().startswith("UNFIXABLE"):
                    es["status"] = "skipped"
                    save_state(state)
                    continue

            # Check prerequisites
            if not prereqs_met(exp_id, state):
                continue

            # Submit!
            script = get_slurm_script(exp_id)
            config_path = get_config_path(exp_id)
            args = [config_path] if config_path else []

            job_id = sbatch(script, args)
            if job_id:
                es["status"] = "submitted"
                es["slurm_job_id"] = job_id
                if es.get("retries", 0) > 0:
                    log.info(f"▶ Re-submitted {exp_id} → job {job_id} (retry {es['retries']})")
                else:
                    log.info(f"▶ Submitted {exp_id} → job {job_id}")
                es["retries"] = es.get("retries", 0) + 1
                n_active += 1
                save_state(state)

        # ── 3. Check if we're done ───────────────────────────────────────
        all_done = True
        n_completed = 0
        n_failed = 0
        n_skipped = 0
        for exp_id in all_experiments:
            if exp_id in VIRTUAL_JOBS:
                continue
            es = get_exp_state(state, exp_id)
            if es["status"] == "completed":
                n_completed += 1
            elif es["status"] in ("failed", "skipped"):
                if es.get("retries", 0) >= MAX_RETRIES or \
                   es.get("last_error", "").upper().startswith("UNFIXABLE"):
                    n_skipped += 1
                else:
                    all_done = False
            else:
                all_done = False

        n_total = len([e for e in all_experiments if e not in VIRTUAL_JOBS])
        log.info(
            f"  Status: {n_completed}/{n_total} completed, "
            f"{n_skipped} skipped, {n_active} active"
        )

        if all_done:
            log.info("")
            log.info("=" * 60)
            log.info("ALL EXPERIMENTS COMPLETE!")
            log.info(f"  Completed: {n_completed}")
            log.info(f"  Skipped/Failed: {n_skipped}")
            log.info("=" * 60)

            # Final aggregation via Claude
            log.info("Running final results aggregation...")
            response, _ = call_claude(
                "All experiments are done. Run `python -m src.evaluation.aggregate_results` "
                "and `python scripts/plot_results.py`. Then give a brief summary of the "
                "overall results — which approach (heuristic vs IFR) worked best, and what "
                "the quality-efficiency trade-off looks like.",
                state.get("session_id"),
            )
            log.info(f"Final summary:\n{response}")
            break

        # ── 4. Sleep ─────────────────────────────────────────────────────
        log.info(f"  Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        run_orchestrator()
    except KeyboardInterrupt:
        log.info("\nOrchestrator interrupted. State saved. Resume by running again.")
    except Exception as e:
        log.exception(f"Orchestrator crashed: {e}")
        log.info("State saved. Resume by running again.")
