"""Dataset generator and loader for Phishing Email Detection."""

from __future__ import annotations

import random
from pathlib import Path
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "emails.csv"

PHISHING_TEMPLATES = [
    (
        "URGENT: Your {brand} account has been suspended",
        "Dear Customer, We detected unauthorized login attempts on your {brand} account. "
        "Please verify your identity immediately by clicking {url} or your account will be permanently locked within 24 hours!",
    ),
    (
        "Action Required: Confirm your {brand} password now",
        "Security Alert! Your {brand} password is about to expire today. "
        "Update your password and billing details here: {url} to avoid service interruption.",
    ),
    (
        "Congratulations! You won a ${amount} {brand} Gift Card",
        "You have been selected as our lucky winner! Claim your ${amount} reward before it expires. "
        "Click here {url} and enter your bank and contact information to receive your prize!",
    ),
    (
        "Unusual sign-in activity detected on your {brand} account",
        "Someone tried to access your {brand} profile from an unrecognized device. "
        "If this was not you, confirm your account credentials immediately at {url}.",
    ),
    (
        "Payment Failed: Update your {brand} billing method",
        "Your recent payment of ${amount} could not be processed. "
        "Your subscription will be suspended unless you verify your card details at {url} right now.",
    ),
    (
        "Final Notice: Tax Refund Pending (${amount})",
        "IRS Notification: You are eligible for a tax refund of ${amount}. "
        "Submit your verification form immediately at {url} to claim your direct deposit.",
    ),
    (
        "IT Helpdesk: Mandatory Email Migration",
        "All employees must re-validate their corporate email password before 5 PM today. "
        "Failure to confirm via {url} will result in mailbox deactivation.",
    ),
    (
        "Package Delivery Failed - Action Needed",
        "We could not deliver your parcel due to an incomplete address. "
        "Pay a small redelivery fee and confirm your address at {url} immediately.",
    ),
    (
        "Your {brand} account is limited",
        "We noticed suspicious transactions on your {brand} wallet. "
        "Restore full access by logging in at {url} and verifying your SSN and password.",
    ),
    (
        "Invoice #{inv} Overdue - Immediate Payment Required",
        "Please find your overdue invoice #{inv} for ${amount}. "
        "Download the statement and settle the balance today at {url} to avoid legal action.",
    ),
]

SAFE_TEMPLATES = [
    (
        "Weekly Team Sync - Agenda for Thursday",
        "Hi team, Here is the agenda for our Thursday standup at 10:00 AM. "
        "We will review sprint progress, blockers, and upcoming release milestones. Notes are in {url}.",
    ),
    (
        "Your {brand} order #{inv} has shipped",
        "Hello, Good news! Your recent order #{inv} has shipped and is expected to arrive on Friday. "
        "You can track your shipment anytime from your {brand} orders page at {url}.",
    ),
    (
        "Monthly Newsletter: Engineering & Product Updates",
        "Welcome to our monthly digest! This month we launched faster search, improved dashboard accessibility, "
        "and published a new blog post on system reliability. Read more at {url}.",
    ),
    (
        "Meeting Notes: Q3 Planning Session",
        "Thanks everyone for joining today's planning session. I have uploaded the summary document "
        "and action items to our shared workspace at {url}. Let me know if anything is missing.",
    ),
    (
        "Receipt for your {brand} subscription",
        "Thank you for your payment of ${amount}. Your monthly {brand} plan has been renewed. "
        "You can view your billing history in your account settings at {url}.",
    ),
    (
        "Reminder: Dentist Appointment Tomorrow at 2:30 PM",
        "Hi, This is a friendly reminder about your scheduled appointment tomorrow at 2:30 PM. "
        "Please reply to this email or call our front desk if you need to reschedule.",
    ),
    (
        "GitHub: Pull request #{inv} merged",
        "The pull request 'Refactor data pipeline and add unit tests' has been reviewed and merged into main. "
        "View the commit diff and CI status on {url}.",
    ),
    (
        "Lunch & Learn: Introduction to Scikit-learn",
        "Join us this Wednesday in Conference Room B for a hands-on workshop on building machine learning models "
        "with Scikit-learn. Slides and sample notebooks are available at {url}.",
    ),
    (
        "Flight Confirmation: Booking #{inv}",
        "Your flight reservation is confirmed. Check-in opens 24 hours before departure. "
        "Review your itinerary and baggage allowance on our official portal: {url}.",
    ),
    (
        "Happy Birthday from the HR Team!",
        "Wishing you a wonderful birthday and a great year ahead! "
        "Don't forget to check out the employee wellness perks page at {url}.",
    ),
]

PHISHING_BRANDS = ["PayPal", "Microsoft", "Amazon", "Netflix", "Chase Bank", "Apple", "Google", "DHL"]
SAFE_BRANDS = ["Amazon", "Spotify", "GitHub", "Coursera", "Notion", "Zoom", "Delta Airlines", "Slack"]

PHISHING_URLS = [
    "http://192.168.45.12/login/verify",
    "http://secure-login-update.xyz/auth",
    "http://bit.ly/3xVerifyNow",
    "http://tinyurl.com/claim-prize-99",
    "http://account-verify.top/confirm?id=882",
    "http://paypal.security-check.ru/login",
    "http://104.21.55.9/webmail/auth",
    "https://login-support.info/reset-password",
    "http://verify-account.club/billing",
    "http://secure@update-portal.tk/signin",
]

SAFE_URLS = [
    "https://github.com/org/repo/pull/412",
    "https://docs.google.com/document/d/notes",
    "https://www.amazon.com/gp/your-account/order-history",
    "https://calendar.google.com/event",
    "https://scikit-learn.org/stable/documentation.html",
    "https://www.notion.so/team/q3-planning",
    "https://support.zoom.us/hc/en-us",
    "https://www.coursera.org/learn/machine-learning",
    "https://slack.com/workspace/updates",
    "https://www.delta.com/mytrips",
]


def generate_dataset(
    output_path: Path = DEFAULT_DATA_PATH,
    num_samples: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a balanced synthetic dataset of Phishing and Safe emails."""
    rng = random.Random(seed)
    half = num_samples // 2
    rows: list[dict[str, str]] = []

    for i in range(half):
        subj_tmpl, body_tmpl = rng.choice(PHISHING_TEMPLATES)
        brand = rng.choice(PHISHING_BRANDS)
        url = rng.choice(PHISHING_URLS)
        amount = rng.choice(["49.99", "250.00", "500", "1,250.00", "99.00"])
        inv = str(rng.randint(1000, 9999))
        subject = subj_tmpl.format(brand=brand, amount=amount, inv=inv)
        body = body_tmpl.format(brand=brand, url=url, amount=amount, inv=inv)
        if i % 4 == 0:
            body += " Act fast before your access is revoked!"
        rows.append({"subject": subject, "body": body, "label": "Phishing"})

    for i in range(num_samples - half):
        subj_tmpl, body_tmpl = rng.choice(SAFE_TEMPLATES)
        brand = rng.choice(SAFE_BRANDS)
        url = rng.choice(SAFE_URLS)
        amount = rng.choice(["9.99", "15.00", "29.00", "45.50"])
        inv = str(rng.randint(100, 999))
        subject = subj_tmpl.format(brand=brand, amount=amount, inv=inv)
        body = body_tmpl.format(brand=brand, url=url, amount=amount, inv=inv)
        if i % 4 == 0:
            body += " Best regards, Team."
        rows.append({"subject": subject, "body": body, "label": "Safe"})

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_dataset(csv_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the email dataset from CSV, generating it first if missing."""
    if not csv_path.exists():
        generate_dataset(csv_path)
    df = pd.read_csv(csv_path)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = df["subject"] + "\n" + df["body"]
    return df


if __name__ == "__main__":
    dataframe = generate_dataset()
    print(f"Generated {len(dataframe)} emails at {DEFAULT_DATA_PATH}")
    print(dataframe["label"].value_counts().to_string())
