from django.test import TestCase, Client
from rest_framework.test import APIClient
from Users.models import ElevationMap, CustomUser, ReliefMap
from django.core.files.uploadedfile import SimpleUploadedFile


class ElevationMapTests(TestCase):

    def setUp(self):
        self.client = APIClient()
        self.user = CustomUser.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)

        # Create a sample ElevationMap for testing
        self.elevation_map = ElevationMap.objects.create(
            user_id=self.user,
            file_path=SimpleUploadedFile("file.tif", b"file_content")
        )

    def test_list_elevation_maps(self):
        response = self.client.get('/path_to_list_elevation_maps_endpoint/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_retrieve_elevation_map(self):
        response = self.client.get(f'/path_to_retrieve_elevation_map_endpoint/{self.elevation_map.id}/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['id'], self.elevation_map.id)

    def test_retrieve_nonexistent_elevation_map(self):
        response = self.client.get('/path_to_retrieve_elevation_map_endpoint/9999/')
        self.assertEqual(response.status_code, 404)

    def test_generate_map(self):
        data = {
            'map_name': 'test_map',
            'macroDisplayedValue': 'macro_value',
            'microDisplayedValue': 'micro_value',
            'illuminationDisplayedValue': 'illumination_value',
            'flatAreasAmountDisplayedValue': 'flat_areas_amount_value',
            'flatAreasSizeDisplayedValue': 'flat_areas_size_value',
            'terrainTypeDisplayedValue': 'terrain_type_value',
            'nnType': 'nn_type_value',
            'aerialPerpectiveDisplayedValue': 'aerial_perspective_value',
            'contrastDisplayedValue': 'contrast_value'
        }
        response = self.client.post('/path_to_generate_map_endpoint/', data)
        self.assertEqual(response.status_code, 201)

    def test_download_map(self):
        data = {
            'dem_type': 'test_dem',
            'south': 10.0,
            'north': 20.0,
            'west': 30.0,
            'east': 40.0,
        }
        response = self.client.post('/path_to_download_map_endpoint/', data)
        self.assertEqual(response.status_code, 200)
