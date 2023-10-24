from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from Users.models import CustomUser
from unittest.mock import patch
from .models import CustomUser


class UserTests(TestCase):

    def setUp(self):
        self.client = APIClient()
        self.user_data = {
            'email': 'testuser@example.com',
        }
        self.user = CustomUser.objects.create(**self.user_data)

    def test_login_user(self):
        url = reverse('loginview')
        response = self.client.post(url, {'token': 'test_token'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('token', response.data)

    def test_purchase_credits(self):
        self.client.force_authenticate(user=self.user)
        url = reverse(
            'purchase_credits')
        response = self.client.post(url, {'amount': 10}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['balance'], 10)

    def test_profile(self):
        self.client.force_authenticate(user=self.user)
        url = reverse('profile')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], self.user_data['email'])

    def test_add_ot_token(self):
        self.client.force_authenticate(user=self.user)
        url = reverse('add_ot_token',
                      args=['test_token'])
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertEqual(self.user.ot_token, 'test_token')

    def test_create_payment_intent(self):
        self.client.force_authenticate(user=self.user)
        url = reverse('create_payment_intent')
        response = self.client.post(url, {'amount': 0.01}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsNotNone(response.data['clientSecret'])

    class CreateCheckOutSessionTests(TestCase):

        def setUp(self):
            self.client = APIClient()
            self.user = CustomUser.objects.create_user(username='testuser', password='testpass')
            self.client.force_authenticate(user=self.user)

        
        def test_create_checkout_session_success(self, mock_create_session):
            # Mock the Stripe Session create method to avoid making real API calls
            mock_create_session.return_value = {
                'id': 'mocked_session_id',
                'url': 'http://mocked_url'
            }

            # Define the URL and payload
            url = '/path_to_create_checkout_session_endpoint/'
            payload = {'quantity': 1}

            # Make the POST request
            response = self.client.post(url, data=payload, format='json')

            # Check the status code and response data
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            self.assertEqual(response.data['id'], 'mocked_session_id')
            self.assertEqual(response.data['url'], 'http://mocked_url')

        
        def test_create_checkout_session_failure(self, mock_create_session):
            # Mock the Stripe Session create method to raise an exception
            mock_create_session.side_effect = Exception('An error occurred')

            # Define the URL and payload
            url = '/path_to_create_checkout_session_endpoint/'
            payload = {'quantity': 1}

            # Make the POST request
            response = self.client.post(url, data=payload, format='json')

            # Check the status code and response data
            self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
            self.assertEqual(response.data['error'], 'An error occurred')
