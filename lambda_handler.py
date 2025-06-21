from mangum import Mangum
from app import app

# Create handler for AWS Lambda for deployment on the AWS cloud.
handler = Mangum(app) 
