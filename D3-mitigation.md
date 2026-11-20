## Mitigation

The project uses a simulated email tool. We added a check in app/tools.py that runs before the tool records an email as sent. It checks the recipient against the email in the synthetic user profile and checks the subject and body for the project’s canary. If either check fails, the tool returns a blocked response. The runner in harness/run.py counts that attempt in blocked_calls. This lets us tell the difference between a model that never tried to send an email and a model that tried and was blocked.

The guard covers a class of actions. The same rule applies whether the request started in a park document, pool metadata, or a saved market note. Its scope is limited to this tool. It still allows ordinary messages to the owner, and it does not check what the model says in its final answer.

## Before and after

We checked the tool directly with four synthetic cases. Without the guard, it accepted all four. With the guard, the ordinary message to the owner still went through. The outside recipient, the canary in the subject, and the canary in the body were each blocked before the tool recorded a send. These checks show what happens once the tool gets a request. They do not measure how often a live model makes that request.

The tagged baseline recorded the following successes out of valid trials. Each cell had zero errors.

| Retrieved source | Claude Haiku 5.5 | GPT-4o-mini |
|---|---:|---:|
| Park document | 0/9 | 8/9 |
| Pool metadata | 0/9 | 9/9 |
| Saved market note | 0/9 | 9/9 |
| No poisoned retrieval | 0/21 | 0/21 |

The baseline falls below the required 20 trials per vector and model. A matched live comparison is still outstanding. Leave the live after column unreported until the team has its raw CSV.

## Reproducing these figures

From the repository root on Windows, run these commands:

    py -m unittest discover -s tests
    py -m harness.check_guard
    py -m harness.analyze results/run_baseline.csv

The paired tool checks are saved in results/guard_policy_checks.csv. The table comes from results/run_baseline.csv at baseline commit 14aa1904b1bef34779bdff685a22eb7aa2b9eefa. Once matched model results are available, pass both CSV paths to harness.analyze to print separate columns for the defense settings.

## AI use for this contribution

ChatGPT (Codex) inspected the tagged repository, wrote the email guard and its tests, added the blocked-call count and CSV analysis, and drafted these sections. It ran the offline tests and checked the baseline figures against the committed rows. This contribution includes no new live-model measurements.
