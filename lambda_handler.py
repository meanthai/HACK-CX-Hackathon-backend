from mangum import Mangum
from app import app

# Create handler for AWS Lambda
handler = Mangum(app) 