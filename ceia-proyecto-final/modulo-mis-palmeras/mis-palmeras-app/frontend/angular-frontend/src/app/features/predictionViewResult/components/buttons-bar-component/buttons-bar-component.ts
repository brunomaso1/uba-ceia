import { Component, output, OutputEmitterRef } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-buttons-bar-component',
  imports: [MatButtonModule],
  templateUrl: './buttons-bar-component.html',
  styleUrl: './buttons-bar-component.scss'
})
export class ButtonsBarComponent {
  exportKMLClicked: OutputEmitterRef<void> = output<void>();
  downloadImageClicked: OutputEmitterRef<void> = output<void>();
  exportZipClicked: OutputEmitterRef<void> = output<void>();

  onExportKMLClick(): void {
    this.exportKMLClicked.emit();
  }

  onDowonloadImageClick(): void {
    this.downloadImageClicked.emit();
  }

  onExportZipClick(): void {
    this.exportZipClicked.emit();
  }
}
