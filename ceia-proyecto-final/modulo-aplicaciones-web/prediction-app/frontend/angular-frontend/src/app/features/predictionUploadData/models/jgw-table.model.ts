import { JGWTableRowModel } from "./jgw-table-row.model";

export class JGWTableModel {
    static generateTableModel(): JGWTableRowModel[] {
        return [
            { value: 0, field: 'Tamaño pixel X' },
            { value: 0, field: 'Ángulo rotación Y' },
            { value: 0, field: 'Ángulo rotación X' },
            { value: 0, field: 'Tamaño pixel Y' },
            { value: 0, field: 'Origen X' },
            { value: 0, field: 'Origen Y' }
        ];
    }
}
