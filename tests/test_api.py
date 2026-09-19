import unittest
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestFlightPriceAPI(unittest.TestCase):
    def setUp(self):
        from flight_price.src.app import app
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        """Test health check route returns 200 OK"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'status': 'healthy'})

    def test_predict_empty_payload(self):
        """Test prediction route handles empty payload gracefully"""
        response = self.app.post('/predict', json={})
        self.assertEqual(response.status_code, 400)

class TestGenderClassificationAPI(unittest.TestCase):
    def setUp(self):
        from gender_classification.src.app import app
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        """Test health check route returns 200 OK"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'status': 'healthy'})

    def test_classify_missing_fields(self):
        """Test classify route returns 400 when fields are missing"""
        response = self.app.post('/classify', json={"age": 25})
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
