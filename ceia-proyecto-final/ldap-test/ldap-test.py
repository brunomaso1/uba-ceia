import ldap
from django_auth_ldap.config import LDAPSearch

def test_ldap_connection(server_url, base_dn, username, password):
    try:
        # Conectar al servidor LDAP
        connection = ldap.initialize(server_url)
        _BASE_DN = base_dn
        _PEOPLE_OU = "OU=people"
        _GROUP_OU = "OU=groups"
        _PEOPLE_DN = f"{_PEOPLE_OU},{_BASE_DN}"
        _GROUP_DN = f"{_GROUP_OU},{_BASE_DN}"
        user_dn = f"CN={username},{_PEOPLE_DN}"
        connection.simple_bind_s(user_dn, password)
        print("Conexión exitosa al servidor LDAP")
        
        # # Realizar una búsqueda de prueba
        # search_filter = "(objectClass=person)"
        # result = connection.search_s(base_dn, ldap.SCOPE_SUBTREE, search_filter)
        
        # # Imprimir los resultados de la búsqueda
        # print("Personas encontradas:")
        # for dn, entry in result:
        #     print(f"DN: {dn}")
        #     for attr, values in entry.items():
        #         print(f"  {attr}: {values}")

        # # Buscar grupos
        # search_filter = "(objectClass=groupOfNames)"
        # result = connection.search_s(base_dn, ldap.SCOPE_SUBTREE, search_filter)

        # print("\nGrupos encontrados:")
        # for dn, entry in result:
        #     print(f"DN del grupo: {dn}")
        #     if "member" in entry:
        #         miembros = entry["member"]
        #         print("Miembros:")
        #         for miembro in miembros:
        #             print(f"  - {miembro.decode('utf-8')}")
        #     else:
        #         print("El grupo no tiene miembros.")

        # # Buscar unidades organizativas
        # search_filter = "(objectClass=organizationalUnit)"
        # result = connection.search_s(base_dn, ldap.SCOPE_SUBTREE, search_filter)

        # print("\nUnidades organizativas encontradas:")
        # for dn, entry in result:
        #     print(f"DN: {dn}")
        #     for attr, values in entry.items():
        #         print(f"  {attr}: {values}")

        # LDAPsearch user Test
        # search = LDAPSearch(_BASE_DN, ldap.SCOPE_SUBTREE, "(uid=%(user)s)" % {'user': 'imm.admin'})
        # search_filter = search.filterstr
        # result = connection.search_s(_BASE_DN, ldap.SCOPE_SUBTREE, search_filter)

        # # Imprimir los resultados de la búsqueda
        # if result:
        #     for dn, entry in result:
        #         print(f"DN: {dn}")
        #         for attr, values in entry.items():
        #             print(f"  {attr}: {values}")

        # LDAPSearch group Test
        # search = LDAPSearch(_BASE_DN, ldap.SCOPE_SUBTREE, "(objectClass=groupOfNames)")
        # search_filter = search.filterstr
        # result = connection.search_s(_BASE_DN, ldap.SCOPE_SUBTREE, search_filter)

        # # Imprimir los resultados de la búsqueda
        # if result:
        #     for dn, entry in result:
        #         print(f"DN: {dn}")
        #         for attr, values in entry.items():
        #             print(f"  {attr}: {values}")

        # Configuración de búsqueda similar a django-auth-ldap
        user_uid = 'imm.admin'
        group_filter = f"(&(objectClass=groupOfUniqueNames)(uniquemember=uid={user_uid},{_PEOPLE_DN}))"


        # Ejecutar la búsqueda de grupos
        result = connection.search_s(_GROUP_DN, ldap.SCOPE_SUBTREE, group_filter)

        # Imprimir los resultados de la búsqueda
        if result:
            print(f"Grupos encontrados para uid={user_uid}:")
            for dn, entry in result:
                print(f"  DN: {dn}")
                for attr, values in entry.items():
                    print(f"    {attr}: {values}")
        else:
            print(f"No se encontraron grupos para uid={user_uid}")
    except ldap.LDAPError as e:
        print(f"Error al conectar al servidor LDAP: {e}")

if __name__ == "__main__":
    # Configuración del servidor LDAP
    server_url = "ldap://localhost:3890"
    base_dn = "DC=localhost,DC=local"
    # Credenciales de prueba
    username = "cvat_bind"
    password = "aCi]$~gn2uAtmpv:"
    
    # Ejecutar la prueba de conexión
    test_ldap_connection(server_url, base_dn, username, password)