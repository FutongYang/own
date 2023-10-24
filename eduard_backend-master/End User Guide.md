# Introduction

Congratulations! Here's a picture of a dog.

![Dog](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/American_Eskimo_Dog_1.jpg/2560px-American_Eskimo_Dog_1.jpg)

# Endpoints:
Here is a current list of all endpoints:

## POST endpoints
~~~
/api/api-login/
/api/api-logout/
/api/api-logout-all/

/api/add_ot_token/<str: token>/

/api/download_map/
/api/generate_map/

/api/stripe-checkout/
~~~

### POST /api/api-login/
Provide the Google ID token as JSON in the request body

Example request body:
```json
{
    "token":"exampletoken"
}
```

Response on success:
```json
{
    "expiry": "2023-09-14T19:29:53.848946+10:00",
    "token": "da499ce7662d2f517d52be40ef8311f104190f4879d7a64cc900c86693f7ed86"
}
```

For each subsequent request after login, the token must be included in the authorization header. 
The key should be prefixed by the string literal "Token", with whitespace separating the two strings
For example:
```json
{
    "Authorization": "Token 121aaaf1320470f494047bbfb2049f8682c001c9"
}
```

### POST /api/api-logout/
Log out the currently logged in user, invalidating the supplied token. Requires no request body, only the authorization header. 

### POST /api/api-logout-all/
Logs out the currently logged in user, but invalidates all tokens for that account. 


### POST /api/add_ot_token/<str: token>/
Save the user's open topography token. 

Example response: 
```json
{
    "message": "OT token added"
}
```
### POST /api/download_map/
Downloads a relief map from open topography

Parameters:
- dem_type (str, optional): type of digital elevation map
- south (float): South coordinate 
- north (float): North coordinate
- west (float): West coordinate
- east (float): East coordinate

Returns:
- status (str): "fail" or "success",
- message (str): "Map downloaded and saved successfully." if the download was successful,
- filename (str): filename of the tif file downloaded

Example Response:
```json
{
    "status": "success",
    "message": "Map downloaded and saved successfully.",
    "filename": "1_1696923968.tif"
}
```

### POST /api/generate_map/
Generate an elevation map based on provided parameters.

Parameters:

- map_name (str): Name of the downloaded elevation map TIF file
- macroDisplayedValue (int, optional): Macro desplayed value for downsampling (between 0 and 100). Defaults to 0.
- microDisplayedValue (int, optional): Display value for Gaussian blur. Defaults to 0.
- illuminationDisplayedValue (int, optional): Clockwise angle (in degrees) of the light, 0 is at the top left. Defaults to 0.
- flatAreasAmountDisplayedValue (int, optional): Flat areas amount. Does nothing. . Defaults to 0.
- flatAreasSizeDisplayedValue (int, optional): Flat areas size. Does nothing. Defaults to 0.
- terrainTypeDisplayedValue (tuple[int, int], optional): Scales the values in the array to be between the two tuple numbers. Defaults to (0,100).
- nnType (int, optional): The type of neural network-current value is 1. Defaults to 1.
- aerialPerpectiveDisplayedValue (int, optional): Scales the perspective depending on the elevation map. Defaults to 100.
- contrastDisplayedValue (int, optional): Scales the contrast, making the images lighter. Defaults to 0.

Returns:
- message (str): "Elevation map generated successfully!" if successful
- url (str): URL of the relief map image

Example Response:
```json
{
    "message": "Elevation map generated successfully!",
    "url": "https://d5xfp7370yhed.cloudfront.net/relief/a931a13a-f4a6-4ab3-9a4d-d664a30d2aa7.jpg"
}
```

### POST /api/stripe-checkout/
Generates a checkout page to buy tokens.

Parameters:
- amount: Amount of tokens to buy

Returns:

Returns:
- id (str): Checkout session ID
- url (str): URL of the login page.
Example Response:
```json
{
  "id": "cs_test_a1M5vhbOy9tFblWUbemFmv2NoeCNTgyLoYCsYMG6nOAQJ37Owhj0nM6IYD",
    "url": "https://checkout.stripe.com/c/pay/cs_test_a1uvQtxTw1nmUlvNZ8wRy5APpqVquoyxM8043u3JNGhuYOlRppu40YhQx5#fidkdWxOYHwnPyd1blpxYHZxWjA0S2poPUlDdTJvbl13THBHXWNIT0ZXQlFgZExRdG5JXEZsQFJLUl1yb11raE1VYzFtQDZxf1FvPH9EaTU3dndId3FtdUJoaUZyazJSVzcxNmE3YTFRRm1oNTVRdHFrUkpWVicpJ2N3amhWYHdzYHcnP3F3cGApJ2lkfGpwcVF8dWAnPyd2bGtiaWBabHFgaCcpJ2BrZGdpYFVpZGZgbWppYWB3dic%2FcXdwYHgl"
}
```

## Setup Method
def setUp(self):
    ...
Purpose: Initialize variables and states before each test method is executed.
Key Actions:
Client Initialization: self.client = APIClient() creates an instance of the API client for making requests.
User Creation: self.user = CustomUser.objects.create_user(...) generates a test user.
User Authentication: self.client.force_authenticate(user=self.user) authenticates the test user for subsequent API requests.


## Test Case: Failed Checkout Session Creation

@patch('stripe.checkout.Session.create')
def test_create_checkout_session_failure(self, mock_create_session):
    ...
Purpose: Ensure the API endpoint handles failures, such as when the Stripe API call fails, gracefully.
Key Components:
Mocking Failure: mock_create_session.side_effect = Exception('An error occurred') simulates an API failure.
API Request: A POST request is made, similar to the previous test case.
Assertions: Validate that the API returns an appropriate error status code and message upon failure.

##  API View: Validate Payment Intent Creation
Purpose: Validate that the API view correctly creates a payment intent and returns the expected response.
Key Actions:
User Authentication: self.client.force_authenticate(user=self.user) ensures the client is authenticated for the test.
API Request: response = self.client.post(url, {'amount': 0.01}, format='json') sends a POST request with a payload to the API endpoint.
Assertions: Validate the response status code and data to ensure the functionality is as expected.




## GET endpoints
We have the following GET endpoints:
~~~
/api/profile/

/api/elevation_maps/
/api/elevation_maps/<int:map_id>/
/api/relief_maps/

/api/token_price/
~~~
### GET /api/profile/
Get the user's credits, email, and open topography token. 

Example response:
```json
{
    "registration_date": "2023-10-02",
    "credits": 80,
    "ot_token": "1db13fgf963c3caed973606a350e6691",
    "email": "example@gmail.com"
}
```

### GET /api/elevation_maps/
Get all elevation maps belonging to the logged in user

Returns a list of objects with fields:
- user_id (int): ID of the user (primary key)
- file_path (str): file path of the elevation map
- creation_date(str): creation date in yyyy-mm-dd format
- deleted (bool): whether the elevation map file has been deleted or not

Example response: 
```json
[
    {
        "user_id": 1,
        "file_path": "1_1696203541.tif",
        "creation_date": "2023-10-02",
        "deleted": false
    },
    {
        "user_id": 1,
        "file_path": "1_1696207076.tif",
        "creation_date": "2023-10-02",
        "deleted": false
    }
]
```

### GET /api/elevation_maps/<int:map_id>/
Get elevation map by the ID

Parameters:

- map_id (int): ID of the elevation map

Returns an object with the following properties:
- elev_id (int) = ID of the elevation map (primary key)
- user_id (int) = ID of the user (primary key)
- credit_cost (int) = amount of credits required to generate the relief map

Example Response:
```json
{
    "user_id": 1,
    "file_path": "1_1696207076.tif",
    "creation_date": "2023-10-02",
    "deleted": false
}
```

### GET /api/relief_maps/
Get all relief maps belonging to the logged in user

Returns list of objects with fields:
- elev_id (int) = ID of the elevation map (primary key)
- user_id (int) = ID of the user (primary key)
- credit_cost (int) = amount of credits required to generate - the relief map 

Example response:
```json
[
    {
        "elev_id": 1,
        "user_id": 1,
        "credit_cost": 10
    }
]
```
## 'GET' Retrieve an elevation map’s details using its ID.
Key Components:
Authentication: Ensures only authenticated users can access the view.
Parameter: map_id is used to retrieve the specific elevation map.
Try/Except Block: Manages potential exceptions, such as the map not being found.
Serializer: Converts the elevation map object into a JSON response.

### GET /api/token_price/
Get the current price of the token.

Returns list of objects with fields:
- price(int): The price of one token, in cents.
Example response:
```json
[
    {
        "price": 1000
    }
]
```

## DELETE

### DELETE api/elevation_maps/<int:map_id>/delete/

Deletes elevation model, given the model ID.



