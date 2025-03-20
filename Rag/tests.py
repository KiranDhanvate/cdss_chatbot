from django.test import TestCase, Client
from django.urls import reverse
import json

# Create your tests here.
class ChatbotTests(TestCase):
    def setUp(self):
        self.client = Client()
        
    def test_index_page_loads(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        
    def test_chat_api_requires_post(self):
        response = self.client.get(reverse('chat_api'))
        self.assertEqual(response.status_code, 405)
        
    def test_chat_api_requires_message(self):
        response = self.client.post(
            reverse('chat_api'),
            json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400) 