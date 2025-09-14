import { JGWType } from "../types/jgw.type";
import { JGWTableRowModel } from "./jgw-table-row.model";
import { JGWTableModel } from "./jgw-table.model";

export class JGWModel implements JGWType {
    x_pixel_size: number = 0;
    y_rotation: number = 0;
    x_rotation: number = 0;
    y_pixel_size: number = 0;
    x_origin: number = 0;
    y_origin: number = 0;

    constructor(init?: Partial<JGWType>) {
        if (init) {
            Object.assign(this, init);
        }
    }

    static fromString(content: string): JGWModel {
        const lines = content.split('\n').filter(line => line.trim() !== '');
        const values = lines.map(line => parseFloat(line.trim()));
        if (values.length !== 6) {
            throw new Error('Invalid JGW content format. Expected 6 values.' + values);
        }
        return new JGWModel({
            x_pixel_size: values[0],
            y_rotation: values[1],
            x_rotation: values[2],
            y_pixel_size: values[3],
            x_origin: values[4],
            y_origin: values[5]
        });
    }

    isEmpty(): boolean {
        return this.x_pixel_size === 0 && this.y_rotation === 0 && this.x_rotation === 0 &&
            this.y_pixel_size === 0 && this.x_origin === 0 && this.y_origin === 0;
    }

    toString(): string {
        return Object.entries(this)
            .map(([_, value]) => `${value}`)
            .join('\n');
    }

    toJSON(): JGWType {
        return {
            x_pixel_size: this.x_pixel_size,
            y_rotation: this.y_rotation,
            x_rotation: this.x_rotation,
            y_pixel_size: this.y_pixel_size,
            x_origin: this.x_origin,
            y_origin: this.y_origin
        };
    }

    toTableModel(): JGWTableRowModel[] {
        const table: JGWTableRowModel[] = JGWTableModel.generateTableModel();
        table[0].value = this.x_pixel_size;
        table[1].value = this.y_rotation;
        table[2].value = this.x_rotation;
        table[3].value = this.y_pixel_size;
        table[4].value = this.x_origin;
        table[5].value = this.y_origin;
        return table;
    }
}
