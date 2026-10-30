
import re

SUSPICIOUS_PHRASES = [
    "verify your account",
    "account suspended",
    "urgent action required",
    "click here immediately",
    "confirm your password",
    "reset your password",
    "claim your prize",
    "you have won",
    "send your bank details",
]

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)


def detect_phishing(email_text):
    if not isinstance(email_text, str):
        raise TypeError("Email text must be a string.")

    if not email_text.strip():
        return "No email text provided", []

    text = email_text.lower()

    reasons = []
    score = 0

    for phrase in SUSPICIOUS_PHRASES:
        if phrase in text:
            score += 1
            reasons.append(f"Suspicious phrase: {phrase}")

    urls = URL_PATTERN.findall(email_text)

    if urls:
        score += 1
        reasons.append(
            f"Contains {len(urls)} URL(s) for review"
        )

    if score >= 2:
        verdict = "Potential phishing"
    elif score == 1:
        verdict = "Suspicious - review required"
    else:
        verdict = "No obvious warning signs detected"

    return verdict, reasons


if __name__ == "__main__":
    print("=== Phishing Email Detection System ===")
    print("Paste your email below.")
    print("Type END on a new line when finished.\n")

    lines = []

    while True:
        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    email = "\n".join(lines)

    try:
        verdict, reasons = detect_phishing(email)

        print(f"\nResult: {verdict}")

        if reasons:
            print("\nWarning signs:")
            for reason in reasons:
                print(f"- {reason}")
        else:
            print("No configured warning signs were detected.")

        print(
            "\nNote: This rule-based result "
            "is not a security guarantee."
        )

    except TypeError as error:
        print(f"Input error: {error}")
