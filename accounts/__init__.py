"""
Application accounts layer for MAEM NeuralEngram.

Users (standard / Alzheimer's), caregivers, caregiver→patient links,
reminders, caregiver alerts, daily retention snapshots and notification
settings — everything the application needs on top of the memory system.
"""

from accounts.store import AccountStore, AccountError  # noqa: F401
