428Project - D2 mock harness

This is a small local document-retrieval test project for CPTS 428 D2. There are five synthetic .txt documents in corpus/; riverside_park.txt contains the test prompt injection. No real user data is used.

Run from a clean checkout
Requires Python 3. No API keys or extra Python packages are needed for mock mode.

git clone https://github.com/tayylortran/428Project.git

cd 428Project

python -m harness.run --mock

On Windows, py may be used instead of python.

The runner reads config/experiment.json, retrieves matching local documents, builds prompts, and uses a predetermined mock response. It prints a summary and saves a CSV in results/ with a numeric success value for each trial (0 or 1). Mock successes are not real AI attack results. Real model APIs are intentionally not part of this D2 test.
