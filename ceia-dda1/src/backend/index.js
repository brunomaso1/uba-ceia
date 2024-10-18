//=======[ Settings, Imports & Data ]==========================================

var PORT = 3000;

var express = require("express");
var app = express();
var utils = require("./mysql-connector");

// to parse application/json
app.use(express.json());
// to serve static files
app.use(express.static("/home/node/app/static/"));

//=======[ Main module code ]==================================================

app.get("/test/", function (req, res, next) {
  respuesta = "Hola, estoy vivo!";
  res.status(200).send(respuesta);
});

app.get("/devices/", function (req, res, next) {
  utils.query("SELECT * FROM Devices", (error, respuesta, fields) => {
    if (error) {
      console.error(error);
      res.status(409).send(error.sqlMessage);
    } else {
      res.status(200).send(respuesta);
    }
  });
});

// POST para agregar un nuveo dispositivo
app.post("/devices/", function (req, res) {
  // Extraigo los datos
  nombre = req.body.name;
  description = req.body.description;
  icon = req.body.icon;
  type = +req.body.type;
  state = req.body.state;
  range = req.body.range;

  // Valido la entrada
  if (!nombre || !description || !icon || typeof type !== "number") {
    return res.status(400).send("La solicitud no es correcta.");
  }

  // Defino la query con placeholders
  let query = "INSERT INTO Devices (name, description, icon, type, state, device_range) VALUES (?, ?, ?, ?, ?, ?)";

  // Dependiendo del tipo, ajustamos el valor de `state` y `range`
  let queryParams;
  if (type === 0) {
    queryParams = [nombre, description, icon, type, false, null];
  } else {
    queryParams = [nombre, description, icon, type, null, range || 0];
  }

  // Ejecutar la query
  utils.query(query, queryParams, (error, results) => {
    if (error) {
      console.error("Error al insertar el dispositivo:", error);
      return res.status(500).send("Error al insertar el dispositivo");
    }
    // Exito
    res.status(201).send({ message: "Se agregó el dispositivo correctamente", id: results.insertId });
  });
});

// PUT para editar un dispositivo existente
app.put("/devices/:id", function (req, res) {
  let id = +req.params.id;
  let nombre = req.body.name;
  let description = req.body.description;
  let icon = req.body.icon;
  let type = +req.body.type;

  // Valido la entrada
  if (!id || !nombre || !description || !icon || typeof type !== "number") {
    return res.status(400).send("La solicitud no es correcta.");
  }

  // Defino la query con placeholders
  let query = "UPDATE Devices SET name = ?, description = ?, icon = ?, type = ? WHERE id = ?";
  let queryParams = [nombre, description, icon, type, id];

  // Ejecutar la query
  utils.query(query, queryParams, (error, results) => {
    if (error) {
      console.error("Error actualizando el dispositivo:", error);
      return res.status(500).send("Error actualizando el dispositivo");
    }
    if (results.affectedRows === 0) {
      return res.status(404).send("Dispositivo no encontrado");
    }
    // Éxito
    res.status(200).send({ message: "Dispositivo actualizado exitosamente" });
  });
});

// PATCH para editar un dispositivo existente
app.patch("/devices/:id", function (req, res) {
  let id = +req.params.id;

  // Validar que el ID sea válido
  if (!id) {
    return res.status(400).send("ID del dispositivo es requerido.");
  }

  // Inicializar un array para almacenar los campos que se actualizarán
  let fields = [];
  let queryParams = [];

  // Comprobamos qué campos han sido enviados y los añadimos a la consulta
  if (req.body.name) {
    fields.push("name = ?");
    queryParams.push(req.body.name);
  }

  if (req.body.description) {
    fields.push("description = ?");
    queryParams.push(req.body.description);
  }

  if (req.body.icon) {
    fields.push("icon = ?");
    queryParams.push(req.body.icon);
  }

  if (typeof req.body.type === "number") {
    fields.push("type = ?");
    queryParams.push(req.body.type);
  }

  if (typeof req.body.state === "boolean" || req.body.state === 0 || req.body.state === 1) {
    fields.push("state = ?");
    queryParams.push(req.body.state);
  }

  if (typeof req.body.device_range === "number") {
    fields.push("device_range = ?");
    queryParams.push(req.body.device_range);
  }

  // Si no se ha enviado ningún campo para actualizar, devolvemos un error
  if (fields.length === 0) {
    return res.status(400).send("No se han enviado campos válidos para actualizar.");
  }

  // Construimos la consulta SQL dinámica para actualizar solo los campos enviados
  let query = `UPDATE Devices SET ${fields.join(", ")} WHERE id = ?`;
  queryParams.push(id); // Agregamos el ID como el último parámetro para la consulta

  // Ejecutar la query
  utils.query(query, queryParams, (error, results) => {
    if (error) {
      console.error("Error actualizando el dispositivo:", error);
      return res.status(500).send("Error actualizando el dispositivo");
    }
    if (results.affectedRows === 0) {
      return res.status(404).send("Dispositivo no encontrado");
    }
    // Éxito
    res.status(200).send({ message: "Dispositivo actualizado exitosamente" });
  });
});

// DELETE para eliminar un dispositivo existente
app.delete("/devices/:id", function (req, res) {
  let id = +req.params.id;

  // Valido la entrada
  if (!id) {
    return res.status(400).send("ID del dispositivo es requerido.");
  }

  // Defino la query
  let query = "DELETE FROM Devices WHERE id = ?";
  let queryParams = [id];

  // Ejecutar la query
  utils.query(query, queryParams, (error, results) => {
    if (error) {
      console.error("Error eliminando el dispositivo:", error);
      return res.status(500).send("Error eliminando el dispositivo");
    }
    if (results.affectedRows === 0) {
      return res.status(404).send("Dispositivo no encontrado");
    }
    // Éxito
    res.status(200).send({ message: "Dispositivo eliminado existosamente" });
  });
});

app.listen(PORT, function (req, res) {
  console.log("NodeJS API running correctly");
});

//=======[ End of file ]=======================================================
