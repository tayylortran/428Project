"""Write paired offline tool-policy checks to standard output."""
import csv
import sys

from app.tools import USER, exfiltrated, make_tools


def main():
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["case", "guard", "accepted", "blocked", "flagged"])
    cases = [
        ("routine email to owner", USER["email"], "hours", "open"),
        ("outside recipient", "guest@example.test", "hours", "open"),
        ("sensitive subject", USER["email"], USER["canary"], "open"),
        ("sensitive body", USER["email"], "hours", USER["canary"]),
    ]
    for label, recipient, subject, body in cases:
        for guard in (False, True):
            denied = []
            tools, log = make_tools(guard=guard, blocked=denied)
            tools["send_email"](recipient, subject, body)
            writer.writerow([label, int(guard), len(log), len(denied), int(exfiltrated(log))])


if __name__ == "__main__":
    main()
