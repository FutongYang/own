# Users/views.py

from .serializers import *
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from .models import CustomUser, Payments
import json
import datetime
from google.oauth2 import id_token
from google.auth.transport import requests
from dotenv import load_dotenv
import os
from knox.views import LoginView as KnoxLoginView
from knox.views import APIView as KnoxAPIView
from rest_framework import permissions
from django.contrib.auth import login
from django.shortcuts import redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from Users.models import CustomUser
from unittest.mock import patch

import stripe

load_dotenv()
stripe.api_key = os.environ.get("STRIPE_API_KEY")
token_price_id = "price_1NxgVSFp7jkXrIuBljlTN7qh"
url = "http://localhost:3000/map"
endpoint_secret = os.environ.get("STRIPE_ENDPOINT_SECRET")


class LoginView(KnoxLoginView):
    permission_classes = (permissions.AllowAny,)

    def post(self, request):
        """
        Logs in the user given their Google ID token, and responds with a session token

            Parameters:
                token (str): Google ID token

            Returns:
                expiry (str): expiry time for the session token
                token (str): session token to be included in the authorization header
                    for future requests 
        """
        # verify request body and get the token
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        token = serializer.data["token"]

        # verify the ID token with google
        try:
            idinfo = id_token.verify_oauth2_token(token, requests.Request(), os.environ.get("CLIENT_ID"))
        except Exception as e:
            return Response({"error": str(e)}, status=500)
        email = idinfo['email']  # retrieve user's email from the token

        user = CustomUser.objects.filter(email=email).exists()  # check if the user already has an account
        if user:
            user = CustomUser.objects.get(email=email)  # retrieve the user if they have an account
        else:
            # create a new account for the user
            user = CustomUser(
                email=email
            )
            user.save()

        # login the user, generating the session token
        login(request, user)
        return super(LoginView, self).post(request, format=None)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def purchase_credits(request):
    """
    Adds the specified amount of credit to the user's account

        Parameters:
            amount (int): The amount of credits to add to the user's account
        Returns:
            Message (str): Success message
            balance (int): The new amount of credits after the transaction
    """
    # verify the request body and get the amount
    user: CustomUser = request.user
    serializer = PurchaseCreditsSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    amount = serializer.data["amount"]

    if user.credits + amount < 0:
        return Response({
            "error": "User does not have enough credits"
        }, status=400)
    user.save()
    return Response({
        "message": "Successfully added " + str(amount) + " credits",
        "balance": user.credits
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile(request):
    """
    Get the currently logged in user's information

        Parameters:
            None
        Returns:
            registration_date (str): registration date in yyyy-mm-dd format
            Credits (int): Amount of credits in the user's account
            ot_token (str): User's opentopography token
            email (str): The user's email
    """
    # get the user using their session token
    user: CustomUser = request.user

    # get and return the user's information
    serializer = UserSerializer(user)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_ot_token(request, token):
    """
    add the OT token specified in the URL

        Parameters:
            token (str): User's open topography token
    """
    # get the user and add their token
    user: CustomUser = request.user
    user.ot_token = token
    user.save()
    return Response({
        "message": "OT token added"
    })


# Code is used to create payment intent; not necessary with checkout
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_payment_intent(request):
    try:
        amount = request.data.get('amount')
        payment_intent = stripe.PaymentIntent.create(
            amount=amount,
            currency="aud",
            payment_method_types=["card"],
        )
        return Response({'clientSecret': payment_intent.client_secret})
    except Exception as e:
        return Response({
            "error": str(e)
        }, status=500)


# Code creates a checkout session and redirects user to use checkout

# Creates Stripe Checkout page.
@permission_classes([IsAuthenticated])
class CreateCheckOutSession(KnoxAPIView):
    def post(self, request):
        """Creates Checkout session.
        Has list of items and starts payment session.
        Args:
            token: OAuth2 Token
            Quantity: Number of items, can be adjusted in payment screen.

        Returns:
            Session.url: URL of checkout page.
        """
        user: CustomUser = request.user
        try:
            quantity = request.data.get('quantity')
            checkout_session = stripe.checkout.Session.create(
                # change URL to homepage
                success_url=url,
                line_items=[
                    {
                        "price": token_price_id,
                        "quantity": quantity,
                        "adjustable_quantity": {"enabled": True},
                    },
                ],
                mode="payment",
                client_reference_id=user.pk
            )

        except Exception as e:
            return Response({
                "error": str(e)
            }, status=500)

        return Response({"id": checkout_session.id, "url": checkout_session.url})


@csrf_exempt
def my_webhook_view(request):
    """Stripe Webhook listener for receiving payments.
    Has a corresponding Webhook which pings the server with changes.

    Args:
        request (JSON Request): What the webhook sends to us.

    Returns:
        None: Updates the user database.
    """
    payload = request.body
    sig_header = request.META['HTTP_STRIPE_SIGNATURE']
    event = None

    try:
        event = stripe.Event.construct_from(
            json.loads(payload), sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        return HttpResponse(status=400)
    except stripe.error.SignatureVerificationError as e:
        print(f'Error verifying webhook signature: {str(e)}')
        return HttpResponse(status=400)

    # Handle the event
    if event.type == 'checkout.session.completed':
        payment_intent = event.data.object  # contains a stripe.PaymentIntent
        # Then define and call a method to handle the successful payment intent.
        # handle_payment_intent_succeeded(payment_intent)
        # print(payment_intent)
        id = payment_intent.id
        cust_pk = payment_intent.client_reference_id
        # print(f'customer id: {cust_pk}')

        # get the quantity from here
        line_items = stripe.checkout.Session.list_line_items(id, limit=5)
        if line_items is not None:
            li_quantity = line_items.data[0].quantity
            # print(li_data)
            # Update the number of tokens here.
            user = CustomUser.objects.get(id=cust_pk)

            user.credits += int(li_quantity)
            user.save()

            # save transaction to database
            payment = Payments(
                user_id=user,
                no_credits=li_quantity
            )
            payment.save()

        else:
            return Response({
                "error": 'nothing was purchased'
            }, status=500)


    else:
        print('Unhandled event type {}'.format(event.type))

    return HttpResponse(status=200)


@api_view(['GET'])
def token_price(request):
    """
    Retrieves the token price.
    The token price is stored in the Stripe dashboard.

        Parameters:
            None
        Returns:
            The price of a single token, in cents.
    """
    stripeToken = stripe.Price.retrieve(token_price_id)
    return Response({"price": int(stripeToken.unit_amount)})



