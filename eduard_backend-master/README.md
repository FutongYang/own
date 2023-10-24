# Eduard Online Backend
## SETUP
There will be specific instructions for installing it on Linux and Windows.
### Windows
Create a python 3.9 virtual environment with `py -3.9 -m venv venv`

Activate the virtual environment with `./venv/Scripts/activate`

Install requirements with `pip install -r requirements.txt`

If using a GPU, follow the installation instructions in https://pytorch.org/ depending on your system.

Install GDAL on Windows by downloading the whl file using this link: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal

Install with `python -m pip install path-to-wheel-file.whl`

Download the model from https://eduard.earth/EduardOnline/Neural%20shading%20PyTorch%20model%20%26%20code.zip and place the contents of the zip file in ./EduardOnline/Users/src

Run `python ./EduardOnline/manage.py makemigrations`

And then `python ./EduardOnline/manage.py migrate`

Repeat the above two commands whenever models are changed to reflect the changes in the database.

### Linux (Debian/Ubuntu)
Clone directory into home, so structure is ~/eduard_backend etc.
Run the following commands to install Python:
~~~
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.9
~~~
Create a python 3.9 virtual environment with the following:
~~~
sudo apt install python3.9-venv
python3.9 -m venv ~/.venvs/backend
~~~
Activate the virtual environment with
`source ~/.venvs/backend/bin/activate`
Verify that the environment is activated by seeing if the left hand side BASH $ has changed.

Check that the python version is 3.9.13 by running
`python -V`

Install requirements with `pip --no-cache-dir install -r requirements.txt`.

If using a GPU, follow the installation instructions in https://pytorch.org/ depending on your system.

Install GDAL on Linux with 
`sudo apt-get install gdal-bin`
an input your sudo code.

Please note that GDAL is a finicky library and can be difficult to install on an env. It may be easier to install it outside of an environment instead.

Additionally, run 
`sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y`
as this is a requirement for CV2 to work.

Download the model from https://eduard.earth/EduardOnline/Neural%20shading%20PyTorch%20model%20%26%20code.zip and place the contents of the zip file in `~/EduardOnline/Users/src`

Run `python ~/EduardOnline/manage.py makemigrations`
to migrate code across to another system

And then `python ~/EduardOnline/manage.py migrate`
to make sure that it is working properly.


### After installation:
Add a .env file to the root directory. The .env file should contain the variable CLIENT_ID, STRIPE_API_KEY and STRIPE_ENDPOINT_SECRET.
which contains the google client id in Discord.
Remember to update with makemigrations and migrate after.
## Running the server
Run with `python ./EduardOnline/manage.py runserver`

You will be able to see http://127.0.0.1:8000/
modify the url to access different endpoints
e.g. http://127.0.0.1:8000/api/api-login/

Run `python ./EduardOnline/manage.py test` to run tests.

To create a webhook, create a stripe listener using the Stripe CLI with the address being 'http://127.0.0.1:8000/api/webhook/'. Then change the line in the .env file:
```
STRIPE_ENDPOINT_SECRET = 
```
to the endpoint secret, listed above. This ensures that the endpoint that is being sent is correct.

## Endpoints
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

### POST /api/add_ot_token/<str: token>/
Save the user's open topography token. 

Example response: 
```json
{
    "message": "OT token added"
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
