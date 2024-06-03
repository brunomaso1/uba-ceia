db.ventasPorProducto.aggregate([
    {
      $group: {
        _id: "$producto",
        ventaPromedioPonderada: {
          $accumulator: {
            init: function() {
              // Inicialización del acumulador
              return { sumaPonderada: 0, contador: 0 };
            },
            accumulate: function(state, ventasDiarias) {
              // Lógica para acumular valores
              let peso = 1.1; // Lógica de negocio
              ventasDiarias.forEach(venta => {
                state.sumaPonderada += venta * peso;
                state.contador += 1;
              });
              return state;
            },
            accumulateArgs: ["$ventas"], // Argumentos para la función accumulate
            merge: function(state1, state2) {
              // Cómo combinar resultados de múltiples documentos
              return {
                sumaPonderada: state1.sumaPonderada + state2.sumaPonderada,
                contador: state1.contador + state2.contador
              };
            },
            finalize: function(state) {
              // Cálculo final para obtener el resultado
              return state.sumaPonderada / state.contador;
            },
            lang: "js"
          }
        }
      }
    }
  ]);
  