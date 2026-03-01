import os
import logging
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("alerts")

class AlertManager:
    """Handles sending email and SMS alerts via SMTP gateway."""
    def __init__(self):
        self.host = os.getenv("SMTP_HOST")
        self.port = 465
        self.user = os.getenv("SMTP_USER")
        self.password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("SMTP_FROM")
        self.to_sms = os.getenv("SMTP_TO_SMS")

    def send_alert(self, subject: str, message: str):
        if not all([self.host, self.user, self.password, self.to_sms]):
            logger.error("Alert config missing in .env. Skipping alert.")
            return
        
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = self.to_sms
        
        try:
            with smtplib.SMTP_SSL(self.host, self.port) as server:
                server.login(self.user, self.password)
                server.send_message(msg)
            logger.info(f"event=ALERT_SENT subject='{subject}' to='{self.to_sms}'")
        except Exception as e:
            logger.error(f"event=ALERT_FAILURE error='{e}'")
