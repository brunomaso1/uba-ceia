import { AfterViewInit, Component, effect, input, InputSignal, signal, WritableSignal } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { FeatureCollection } from 'geojson';
import * as L from 'leaflet'

@Component({
  selector: 'app-map-component',
  imports: [MatCardModule],
  templateUrl: './map-component.html',
  styleUrl: './map-component.scss'
})
export class MapComponent implements AfterViewInit {
  geoJSONData: InputSignal<FeatureCollection | undefined> = input<FeatureCollection>();
  markers: L.Marker[] = [];

  private readonly baseMapURl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
  private readonly baseMapOptions = {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
  };

  private map!: L.Map;
  private currentGeoJsonLayer: L.GeoJSON | undefined;

  constructor() {
    effect(() => {
      const geoJSON = this.geoJSONData();

      if (this.map && geoJSON) {
        this.addGeoJSONLayer(geoJSON);
      }
    });
    L.Icon.Default.imagePath = "assets/leaflet/"
  }

  ngAfterViewInit(): void {
    this.initMap();
    // this.centerMap();
  }

  private initMap(): void {
    this.map = L.map('map')
    this.map.setView([-34.897735, -56.164601], 13); // Set initial view to Montevideo, Uruguay
    L.tileLayer(this.baseMapURl, this.baseMapOptions).addTo(this.map);

    const initialLayer = this.geoJSONData();
    if (initialLayer) {
      this.addGeoJSONLayer(initialLayer);
    }
  }

  private centerMap(): void {
    // Create a boundary based on the markers
    const bounds = L.latLngBounds(this.markers.map(marker => marker.getLatLng()));

    // Fit the map into the boundary
    this.map.fitBounds(bounds);
  }

  private addGeoJSONLayer(geoJSON: FeatureCollection): void {
    if (!this.map) {
      console.warn('Aún no se ha inicializado el mapa. No se ha podido agregar la capa GeoJSON.');
      return;
    }

    if (this.currentGeoJsonLayer) {
      this.map.removeLayer(this.currentGeoJsonLayer);
    }

    const greenIcon = L.icon({
      iconUrl: '/assets/iconos-palmeras/palmera-verde.png',
      shadowUrl: '/assets/leaflet/marker-shadow.png',

      iconSize: [40, 40], // size of the icon
      shadowSize: [80, 40], // size of the shadow
      iconAnchor: [23, 40], // point of the icon which will correspond to marker's location
      shadowAnchor: [20, 40],  // the same for the shadow
      popupAnchor: [-5, -35] // point from which the popup should open relative to the iconAnchor
    });

    const geoJsonLayer = L.geoJSON(geoJSON, {
      pointToLayer: (feature, latlng) => {
        switch (feature.properties?.name) {
          case 'palmera-sana':
            return L.marker(latlng, {
              icon: greenIcon
            });
          default:
            return L.marker(latlng);
        }
      },
      onEachFeature: (feature, layer) => {
        if (feature.properties?.name) {
          layer.bindPopup(feature.properties.name);
        }
      }
    });

    geoJsonLayer.addTo(this.map);
    this.currentGeoJsonLayer = geoJsonLayer;
    this.map.fitBounds(geoJsonLayer.getBounds());
  }
}