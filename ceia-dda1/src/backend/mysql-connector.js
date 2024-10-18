//=======[ Settings, Imports & Data ]==========================================
// Se modifica archivo de conexión porque no estaba cerrando bien la conección y daba error 409.
// https://stackoverflow.com/questions/54888061/configuration-mysql-node-js-error-can-not-enqueue-query-after-fatal-error
var mysql = require('mysql');

var pool = mysql.createPool({
    host     : 'mysql-server',
    port     : '3306',
    user     : 'root',
    password : 'userpass',
    database : 'smart_home'
});

//=======[ Main module code ]==================================================

module.exports = {
    query: function(){
        var sql_args = [];
        var args = [];
        for(var i=0; i<arguments.length; i++){
            args.push(arguments[i]);
        }
        var callback = args[args.length-1]; //last arg is callback
        pool.getConnection(function(err, connection) {
        if(err) {
                console.log(err);
                return callback(err);
            }
            if(args.length > 2){
                sql_args = args[1];
            }
        connection.query(args[0], sql_args, function(err, results) {
          connection.release(); // always put connection back in pool after last query
          if(err){
                    console.log(err);
                    return callback(err);
                }
          callback(null, results);
        });
      });
    }
};

//=======[ End of file ]=======================================================



