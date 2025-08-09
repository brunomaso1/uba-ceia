import { NgOptimizedImage } from '@angular/common';
import { Component } from '@angular/core';

@Component({
  selector: 'app-image-viewer-component',
  imports: [NgOptimizedImage],
  templateUrl: './image-viewer-component.html',
  styleUrl: './image-viewer-component.scss'
})
export class ImageViewerComponent {
  imageUrl: string = '/assets/images/img3.jpg';
  zoomLevel: number = 1;

  isDragging: boolean = false;
  lastMouseX: number = 0;
  lastMouseY: number = 0;
  translateX: number = 0;
  translateY: number = 0;

  zoomIn(): void {
    this.zoomLevel += 0.1;
  }

  zoomOut(): void {
    this.zoomLevel -= this.zoomLevel > 0.05 ? 0.1 : 0;
    if (this.zoomLevel < 0.1) {
      this.translateX = 0;
      this.translateY = 0;
    }
  }

  startDrag(event: MouseEvent): void {
    this.isDragging = true;
    this.lastMouseX = event.clientX;
    this.lastMouseY = event.clientY;
  }

  onDrag(event: MouseEvent): void {
    if (!this.isDragging) return;

    const deltaX = event.clientX - this.lastMouseX;
    const deltaY = event.clientY - this.lastMouseY;

    this.translateX += deltaX;
    this.translateY += deltaY;

    this.lastMouseX = event.clientX;
    this.lastMouseY = event.clientY;
  }

  endDrag(): void {
    this.isDragging = false;
  }

  getTransform(): string {
    return `scale(${this.zoomLevel}) translate(${this.translateX / this.zoomLevel}px, ${this.translateY / this.zoomLevel}px)`;
  }

  onWheel(event: WheelEvent) {
    event.preventDefault();
    const delta = event.deltaY;

    // Hacemos zoom con pasos pequeños
    const zoomFactor = 0.1;
    const newZoom = this.zoomLevel - (delta > 0 ? zoomFactor : -zoomFactor);

    this.zoomLevel = Math.max(0.05, Math.min(newZoom, 5)); // Limita entre 0.1 y 5

    if (this.zoomLevel <= 1) {
      this.translateX = 0;
      this.translateY = 0;
    }
  }
  
  resetTransform() {
    this.zoomLevel = 1;
    this.translateX = 0;
    this.translateY = 0;
    this.isDragging = false;
  }
}