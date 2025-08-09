import { Component, input, InputSignal, output, OutputEmitterRef } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-prediction-buttons-bar-component',
  imports: [MatButtonModule],
  templateUrl: './prediction-buttons-bar-component.html',
  styleUrl: './prediction-buttons-bar-component.scss'
})
export class PredictionButtonsBarComponent {
  predictionButonClicked: OutputEmitterRef<void> = output<void>();
  dissableButton: InputSignal<boolean> = input<boolean>(false);

  onClick(): void {
    this.predictionButonClicked.emit();
  }

  isButtonDisabled(): boolean {
    return this.dissableButton() || false;
  }
}
