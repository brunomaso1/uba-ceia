# Plantilla memoria

Plantilla para la elaboración de la Memoria del Trabajo Final de cualquiera de las Carreras de Especialización o Maestrías que se dictan en el Laboratorio de Sistemas Embebidos en la Facultad de Ingeniería de la UBA.  

Leer [Instructivo para utilizar la plantilla en Overleaf](https://github.com/TTFA-TTFB/Plantilla-para-memoria/blob/main/Instructivo%20Overleaf.md).

## Colores del proyecto

```css
html {
	--mat-sys-background: light-dark(#f9faf3, #121410);
	--mat-sys-error: light-dark(#ba1a1a, #ffb4ab);
	--mat-sys-error-container: light-dark(#ffdad6, #93000a);
	--mat-sys-inverse-on-surface: light-dark(#f1f1eb, #2f312d);
	--mat-sys-inverse-primary: light-dark(#02e600, #026e00);
	--mat-sys-inverse-surface: light-dark(#2f312d, #e2e3dc);
	--mat-sys-on-background: light-dark(#1a1c18, #e2e3dc);
	--mat-sys-on-error: light-dark(#ffffff, #690005);
	--mat-sys-on-error-container: light-dark(#93000a, #ffdad6);
	--mat-sys-on-primary: light-dark(#ffffff, #013a00);
	--mat-sys-on-primary-container: light-dark(#015300, #77ff61);
	--mat-sys-on-primary-fixed: light-dark(#002200, #002200);
	--mat-sys-on-primary-fixed-variant: light-dark(#015300, #015300);
	--mat-sys-on-secondary: light-dark(#ffffff, #263422);
	--mat-sys-on-secondary-container: light-dark(#3c4b37, #d7e8cd);
	--mat-sys-on-secondary-fixed: light-dark(#121f0e, #121f0e);
	--mat-sys-on-secondary-fixed-variant: light-dark(#3c4b37, #3c4b37);
	--mat-sys-on-surface: light-dark(#1a1c18, #e2e3dc);
	--mat-sys-on-surface-variant: light-dark(#43483f, #dfe4d7);
	--mat-sys-on-tertiary: light-dark(#ffffff, #013a00);
	--mat-sys-on-tertiary-container: light-dark(#015300, #77ff61);
	--mat-sys-on-tertiary-fixed: light-dark(#002200, #002200);
	--mat-sys-on-tertiary-fixed-variant: light-dark(#015300, #015300);
	--mat-sys-outline: light-dark(#73796e, #8d9387);
	--mat-sys-outline-variant: light-dark(#c3c8bc, #43483f);
	--mat-sys-primary: light-dark(#026e00, #02e600);
	--mat-sys-primary-container: light-dark(#77ff61, #015300);
	--mat-sys-primary-fixed: light-dark(#77ff61, #77ff61);
	--mat-sys-primary-fixed-dim: light-dark(#02e600, #02e600);
	--mat-sys-scrim: light-dark(#000000, #000000);
	--mat-sys-secondary: light-dark(#54634d, #bbcbb2);
	--mat-sys-secondary-container: light-dark(#d7e8cd, #3c4b37);
	--mat-sys-secondary-fixed: light-dark(#d7e8cd, #d7e8cd);
	--mat-sys-secondary-fixed-dim: light-dark(#bbcbb2, #bbcbb2);
	--mat-sys-shadow: light-dark(#000000, #000000);
	--mat-sys-surface: light-dark(#f9faf3, #121410);
	--mat-sys-surface-bright: light-dark(#f9faf3, #383a35);
	--mat-sys-surface-container: light-dark(#eeeee7, #1e201c);
	--mat-sys-surface-container-high: light-dark(#e8e9e1, #282b26);
	--mat-sys-surface-container-highest: light-dark(#e2e3dc, #333531);
	--mat-sys-surface-container-low: light-dark(#f3f4ed, #1a1c18);
	--mat-sys-surface-container-lowest: light-dark(#ffffff, #0c0f0b);
	--mat-sys-surface-dim: light-dark(#dadbd3, #121410);
	--mat-sys-surface-tint: light-dark(#026e00, #02e600);
	--mat-sys-surface-variant: light-dark(#dfe4d7, #43483f);
	--mat-sys-tertiary: light-dark(#026e00, #02e600);
	--mat-sys-tertiary-container: light-dark(#77ff61, #015300);
	--mat-sys-tertiary-fixed: light-dark(#77ff61, #77ff61);
	--mat-sys-tertiary-fixed-dim: light-dark(#02e600, #02e600);
	--mat-sys-neutral-variant20: #2c3229;
	--mat-sys-neutral10: #1a1c18;
}
```

## Reducir tamaño PDF

Para reducir el tamaño del PDF generado, se puede utilizar la herramienta `ghostscript`. A continuación se muestra un comando de ejemplo para comprimir el archivo PDF:

```powershell
gswin64c -sDEVICE=pdfwrite -o converted.pdf -dCompatibilityLevel='1.4' -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH memorianueva.pdf
```
Los valores posibles son:
- /screen → máxima compresión (menor calidad)
- /ebook → buena calidad (recomendado)
- /printer → alta calidad
- /prepress → casi sin compresión