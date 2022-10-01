from email.mime.text import MIMEText
from smtplib import SMTPException
import smtplib
from email.header import Header

smtp_host = 'smtp.gmail.com'
smtp_port = 465
username = 'kalashnikova1120@gmail.com'
password = 
from_address = 'kalashnikova1120@gmail.com'
to_address = 'kalashnikova1120@gmail.com'
charset = 'utf-8'
subject = 'test subject'
body = 'test'
smtp = smtplib.SMTP_SSL(smtp_host, smtp_port)
smtp.login(username, password)

#message = ("From: %srnTo: %srnSubject: %srnrn%s" % (from_address, to_address, subject, body))
message = MIMEText(body, 'plain', charset)
message['Subject'] = Header(subject, charset)
message['From'] = from_address
message['To'] = to_address

#result = smtp.sendmail(from_address, to_address, message)
smtp.sendmail(from_address, [to_address],
             message.as_string())
smtp.close()
