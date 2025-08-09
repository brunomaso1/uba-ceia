import { NgOptimizedImage } from '@angular/common';
import { Component, inject, OnDestroy, Signal, signal, WritableSignal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-upload-image-component',
  imports: [MatButtonModule, MatCardModule],
  templateUrl: './upload-image-component.html',
  styleUrl: './upload-image-component.scss'
})
export class UploadImageComponent implements OnDestroy {
  fileName: string = '';
  selectedFile: File | null = null;
  imageUrl: SafeUrl | null = null;
  sanitizer: DomSanitizer = inject(DomSanitizer);
  isImageTooLarge: WritableSignal<boolean> = signal(false);

  onFileSelected(event: Event): void {
    const file: File | undefined = (event.target as HTMLInputElement).files?.[0];

    if (file) {
      this.selectedFile = file;
      this.fileName = file.name;

      const MAX_PREVIEW_SIZE_MB = 50;
      if (file.size <= MAX_PREVIEW_SIZE_MB * 1024 * 1024) {
        // Show the image preview
        const reader = new FileReader();
        reader.onload = (e: ProgressEvent<FileReader>) => {
          if (reader.result) {
            this.imageUrl = this.sanitizer.bypassSecurityTrustUrl(reader.result as string); // Trust the URL for safe use in the template (Angular needs this for security reasons)
          }
        }
        reader.readAsDataURL(file); // Encodes the file as a base64 string
      } else {
        alert(`El archivo es demasiado grande para visualizar en el navegador. Sin embargo, se ha cargado correctamente.`);
        this.isImageTooLarge.set(true);
      }
    }
  }

  ngOnDestroy(): void {
    this.imageUrl = null;
  }

  getImage(): File | null {
    return this.selectedFile;
  }
}