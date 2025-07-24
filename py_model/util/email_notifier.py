import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailNotifier:
    def __init__(self, sender_email, sender_password):
        self.smtp_server = 'smtp.gmail.com'
        self.port = 465
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = 'asho227@naver.com'  # 여기를 고정 수신자로 지정

    def send_alert(self, subject='🚨 LED 상태 경고', body='빨간 LED가 감지되었습니다.'):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP_SSL(self.smtp_server, self.port) as server:
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            print("📧 이메일 전송 성공")
        except Exception as e:
            print(f"❌ 이메일 전송 실패: {e}")