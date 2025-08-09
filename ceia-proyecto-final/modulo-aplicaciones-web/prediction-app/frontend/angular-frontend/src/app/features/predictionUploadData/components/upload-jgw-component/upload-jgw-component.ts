import { Component, effect, signal, WritableSignal } from '@angular/core';
import { JGWModel } from '../../models/jgw.model';
import { MatButtonModule } from '@angular/material/button';
import { MatTableModule } from '@angular/material/table';
import { MatCardModule } from '@angular/material/card';
import { JGWTableRowModel } from '../../models/jgw-table-row.model';
import { JGWTableModel } from '../../models/jgw-table.model';

@Component({
  selector: 'app-upload-jgw-component',
  imports: [MatButtonModule, MatTableModule, MatCardModule],
  templateUrl: './upload-jgw-component.html',
  styleUrl: './upload-jgw-component.scss'
})
export class UploadJGWComponent {
  jgwModel: WritableSignal<JGWModel> = signal(new JGWModel());
  displayedColumns: string[] = ['value', 'field'];
  dataSource: JGWTableRowModel[];

  constructor() {
    this.dataSource = JGWTableModel.generateTableModel();
    effect(() => {
      // Update the data source whenever the JGW model changes
      this.dataSource = this.jgwModel().toTableModel();
    });
  }

  onFileSelected(event: Event) {
    const file: File | undefined = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        const content = reader.result as string;
        const parsedJGW = JGWModel.fromString(content);
        this.jgwModel.set(parsedJGW);
      }
      reader.readAsText(file);
    }
  }

  isEmptyJGW(): boolean {
    return this.jgwModel().isEmpty();
  }

  printJGW(): string {
    return this.jgwModel().toString();
  }

  getJGWModel(): JGWModel | null {
    return this.isEmptyJGW() ? null : this.jgwModel();
  }
}