declare const M;
class Main implements EventListenerObject {
  devices: Device[] = null;
  sucessHtml = `<span class="sucess">Operación efectuada correctamente<span>`;
  wrrongHtml = `<span class="alert">Operación con error<span>`;

  constructor() {
    this.loadDevicesFromDB();

    const cancelButton = <HTMLButtonElement>document.getElementById("dismissModal");
    const acceptButton = <HTMLButtonElement>document.getElementById("sendModal");

    cancelButton.addEventListener("click", () => {
      const addDeviceModal = <HTMLDialogElement>document.getElementById("add-update-device-modal");
      const modalInstance = M.Modal.getInstance(addDeviceModal);
      modalInstance.close();

      // Limpiar el formulario
      const form = <HTMLFormElement>document.getElementById("device-form");
      form.reset();
    });

    acceptButton.addEventListener("click", (object: Event) => {
      this.handleAddUpdateDevicesEvent(object);
    });
  }

  loadDevicesFromDB(): void {
    const xmlHttp = new XMLHttpRequest();

    xmlHttp.onload = () => {
      if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
        this.devices = JSON.parse(xmlHttp.responseText);
        this.renderDevices();
      } else {
        console.error(`HTTP error! status: ${xmlHttp.status}`);
      }
    };

    xmlHttp.onerror = () => {
      console.error("Request failed");
    };

    xmlHttp.open("GET", "http://localhost:8000/devices", true);
    xmlHttp.setRequestHeader("Content-Type", "application/json");
    xmlHttp.send();
  }

  renderDevices() {
    const devices = this.devices;
    let hmtlInterno: string = "";
    for (const device of devices) {
      hmtlInterno += this.createDevice(device);
    }
    const devicesSecction: HTMLElement = document.getElementById("lista-dispositivos");
    let html = `<div class="row devices-list">
      <ul>${hmtlInterno}</ul>
  </div>`;
    devicesSecction.innerHTML = html;

    // Asigna el evento "click" para el botón EDITAR
    const editButtons = document.querySelectorAll(".edit-device-btn");

    for (const button of editButtons) {
      button.addEventListener("click", this.editDevice.bind(this));
    }

    // Asigno el evento "click" para el botón ELIMINAR
    const deleteButtons = document.querySelectorAll(".delete-device-btn");

    for (const button of deleteButtons) {
      button.addEventListener("click", this.deleteDevice.bind(this));
    }

    // Asigno el evento "click" para el botón switch
    const switchButtons = document.querySelectorAll(".switch-btn");

    for (const button of switchButtons) {
      button.addEventListener("click", this.switchButtonPressed.bind(this));
    }

    // Asigno el evento "click" para el botón range
    const rangeButtons = document.querySelectorAll(".device_range-btn");

    for (const button of rangeButtons) {
      button.addEventListener("click", this.deviceRangePressed.bind(this));
    }
  }

  createDevice(device: Device) {
    let deviceHTML = `<div class="col s12 m6 l3">
  <li>
    <div class="card small indigo darken-3 white-text">
      <div class="card-content">
        <span class="card-title">
          <i class="material-icons left">${device.icon}</i><br />
          <strong>${device.name}</strong>
        </span>
        <div class="card-body">
          <p>${device.description}</p>`;
    if (device.type == 0) {
      deviceHTML += `
      <div class="switch">
        <label>
          Off
          <input data-id="${device.id}" class="switch-btn" type="checkbox" ${device.state ? "checked" : ""}>
          <span class="lever"></span>
          On
        </label>
      </div>`;
    } else {
      deviceHTML += `
      <div class="range-field">
        <input data-id="${device.id}" class="device_range-btn" type="range" min="0" max="100" value="${device.device_range}">
      </div>`;
    }
    deviceHTML += `</div>
    </div>
      <div class="card-action">
        <a class="white-text edit-device-btn" href="#" data-id="${device.id}">EDITAR</a>
        <a class="white-text delete-device-btn" href="#" data-id="${device.id}">ELIMINAR</a>
      </div>
    </div>
  </li>
</div>`;
    return deviceHTML;
  }

  switchButtonPressed(event: Event): void {
    const target = <HTMLInputElement>event.target;
    const deviceId = +target.getAttribute("data-id"); // Obtener el ID del dispositivo
    const newState = target.checked; // Obtener el nuevo estado (on/off)

    // Encontrar el dispositivo a actualizar
    const device = this.devices.find((d) => d.id === deviceId);

    device.state = newState;

    this.patchDeviceDB(deviceId, device);
  }

  deviceRangePressed(event: Event): void {
    const target = <HTMLInputElement>event.target;
    const deviceId = +target.getAttribute("data-id"); // Obtener el ID del dispositivo
    const newValue = +target.value; // Obtener el nuevo valor

    // Encontrar el dispositivo a actualizar
    const device = this.devices.find((d) => d.id === deviceId);

    device.device_range = newValue;

    this.patchDeviceDB(deviceId, device);
  }

  handleAddUpdateDevicesEvent(object: Event): void {
    object.preventDefault();
    const form = <HTMLFormElement>document.getElementById("device-form");

    const deviceId = (<HTMLInputElement>document.getElementById("device-id")).value;
    const name = (<HTMLInputElement>document.getElementById("device-name")).value;
    const description = (<HTMLInputElement>document.getElementById("device-description")).value;
    const icon = (<HTMLSelectElement>document.getElementById("device-icon")).value;
    const type = parseInt((<HTMLSelectElement>document.getElementById("device-type")).value);

    let validationError = "";

    // Validaciones
    switch (true) {
      case !name:
        validationError = "El nombre no puede ser vacío";
        break;
      case !description:
        validationError = "La descripción no puede ser vacía";
        break;
      case !icon:
        validationError = "Debe seleccionar un ícono";
        break;
      case isNaN(type):
        validationError = "Debe seleccionar un tipo de dispositivo";
        break;
      default:
        validationError = ""; // No hay errores
    }

    // Si hay un error de validación, mostrar el mensaje con un toast y salir
    if (validationError) {
      M.toast({ html: validationError, classes: "alert" });
      return;
    }

    let device;
    if (type === 0) {
      device = {
        name,
        description,
        icon,
        type,
        state: false,
      };
    } else {
      device = {
        name,
        description,
        icon,
        type,
        device_range: 50,
      };
    }

    if (deviceId) {
      // Si estamos editando, actualizar el dispositivo existente
      this.updateDeviceDB(deviceId, device);
    } else {
      // Si no hay ID, estamos creando un nuevo dispositivo
      this.addDeviceDB(device);
    }

    // Cerrar el modal
    const addDeviceModal = <HTMLDialogElement>document.getElementById("add-update-device-modal");
    const modalInstance = M.Modal.getInstance(addDeviceModal);
    modalInstance.close();

    // Limpiar el formulario
    form.reset();
    // Le asignamos el valor nulo a si estamos editando o no.
    (<HTMLInputElement>document.getElementById("device-id")).value = null;
  }

  editDevice(event: Event): void {
    const target = <HTMLAnchorElement>event.target;
    const deviceId = +target.getAttribute("data-id");

    // Encontrar el dispositivo a editar
    const device = this.devices.find((d) => d.id === deviceId);

    // Rellenar el formulario con los datos del dispositivo
    (<HTMLInputElement>document.getElementById("device-name")).value = device.name;
    (<HTMLInputElement>document.getElementById("device-description")).value = device.description;
    (<HTMLSelectElement>document.getElementById("device-icon")).value = device.icon.toString();
    (<HTMLSelectElement>document.getElementById("device-type")).value = device.type.toString();

    // Guardar el ID del dispositivo que se está editando
    (<HTMLInputElement>document.getElementById("device-id")).value = device.id.toString();

    // Actualizar los labels y los select con Materialize
    M.updateTextFields(); // Actualizar los campos de texto
    M.FormSelect.init(document.querySelectorAll("select"));

    // Abrir el modal de edición
    const addDeviceModal = <HTMLDialogElement>document.getElementById("add-update-device-modal");
    const modalInstance = M.Modal.getInstance(addDeviceModal);
    modalInstance.open();
  }

  deleteDevice(event: Event): void {
    const target = <HTMLAnchorElement>event.target;
    const deviceId = +target.getAttribute("data-id");

    // Encontrar el dispositivo a eliminar
    const device = this.devices.find((d) => d.id === deviceId);

    if (window.confirm(`Desea eliminar el dispositivo ${device.name}?`)) this.deleteDeviceDB(device.id, device);
  }

  deleteDeviceDB(deviceId: number, device: Device): void {
    const xmlHttp = new XMLHttpRequest();

    xmlHttp.onload = () => {
      if (xmlHttp.readyState == 4 && (xmlHttp.status == 200 || xmlHttp.status == 204)) {
        // Si el dispositivo se actualizó correctamente, recarga la lista
        this.loadDevicesFromDB();
        M.toast({ html: this.sucessHtml });
      } else {
        console.error(`HTTP error! status: ${xmlHttp.status}`);
        M.toast({ html: this.wrrongHtml });
      }
    };

    xmlHttp.onerror = () => {
      console.error("Request failed");
    };

    xmlHttp.open("DELETE", `http://localhost:8000/devices/${deviceId}`, true);
    xmlHttp.setRequestHeader("Content-Type", "application/json");
    xmlHttp.send(JSON.stringify(device));
  }

  updateDeviceDB(deviceId: string, device: Device): void {
    const xmlHttp = new XMLHttpRequest();

    xmlHttp.onload = () => {
      if (xmlHttp.readyState == 4 && (xmlHttp.status == 200 || xmlHttp.status == 204)) {
        // Si el dispositivo se actualizó correctamente, recarga la lista
        this.loadDevicesFromDB();
        M.toast({ html: this.sucessHtml });
      } else {
        console.error(`HTTP error! status: ${xmlHttp.status}`);
        M.toast({ html: this.wrrongHtml });
      }
    };

    xmlHttp.onerror = () => {
      console.error("Request failed");
    };

    xmlHttp.open("PUT", `http://localhost:8000/devices/${deviceId}`, true);
    xmlHttp.setRequestHeader("Content-Type", "application/json");
    xmlHttp.send(JSON.stringify(device));
  }

  addDeviceDB(device: Device): void {
    const xmlHttp = new XMLHttpRequest();

    xmlHttp.onload = () => {
      if (xmlHttp.readyState == 4 && xmlHttp.status == 201) {
        this.loadDevicesFromDB(); // Recargar dispositivos
        M.toast({ html: this.sucessHtml });
      } else {
        console.error(`HTTP error! status: ${xmlHttp.status}`);
        M.toast({ html: this.wrrongHtml });
      }
    };

    xmlHttp.onerror = () => {
      console.error("Request failed");
    };

    xmlHttp.open("POST", "http://localhost:8000/devices", true);
    xmlHttp.setRequestHeader("Content-Type", "application/json");
    xmlHttp.send(JSON.stringify(device));
  }

  patchDeviceDB(deviceId: number, device: Device): void {
    const xmlHttp = new XMLHttpRequest();

    xmlHttp.onload = () => {
      if (xmlHttp.readyState == 4 && (xmlHttp.status == 200 || xmlHttp.status == 204)) {
        // Si el dispositivo se actualizó correctamente, recarga la lista
        this.loadDevicesFromDB();
      } else {
        console.error(`HTTP error! status: ${xmlHttp.status}`);
        M.toast({ html: this.wrrongHtml });
      }
    };

    xmlHttp.onerror = () => {
      console.error("Request failed");
    };

    xmlHttp.open("PATCH", `http://localhost:8000/devices/${deviceId}`, true);
    xmlHttp.setRequestHeader("Content-Type", "application/json");
    xmlHttp.send(JSON.stringify(device));
  }

  handleEvent(object: Event): void {
    throw new Error("Not implemented");
  }
}

window.addEventListener("load", () => {
  let main: Main = new Main();

  M.FormSelect.init(document.querySelectorAll("select"), null);
  M.Modal.init(document.querySelectorAll(".modal"), null);
});
