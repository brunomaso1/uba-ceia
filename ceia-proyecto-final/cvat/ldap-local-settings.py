# Overlaying production
from cvat.settings.production import *

# Custom code below
import ldap
from django_auth_ldap.config import LDAPSearch
from django_auth_ldap.config import NestedActiveDirectoryGroupType, GroupOfNamesType, GroupOfUniqueNamesType

# # Activate LDAP debugging
# import logging, logging.handlers
# logfile = "/tmp/django-ldap-debug.log"
# my_logger = logging.getLogger("django_auth_ldap")
# my_logger.setLevel(logging.DEBUG)
# handler = logging.handlers.RotatingFileHandler(logfile, maxBytes=1024 * 500, backupCount=5)
# my_logger.addHandler(handler)

try:
    CVAT_LDAP_USER = os.environ["CVAT_LDAP_USER"]
    CVAT_LDAP_PASSWORD = os.environ["CVAT_LDAP_PASSWORD"]
except KeyError:
    print("Environment variable does not exist")

# Notify CVAT that we are using LDAP authentication
IAM_TYPE = "LDAP"

# Talking to the LDAP server
AUTH_LDAP_SERVER_URI = "ldap://lldap:3890"  # IP Addresses also work
ldap.set_option(ldap.OPT_REFERRALS, 0)

_BASE_DN = "DC=localhost,DC=local"
_PEOPLE_OU = "OU=people"
_GROUP_OU = "OU=groups"
_PEOPLE_DN = "%s,%s" % (_PEOPLE_OU, _BASE_DN)
_GROUP_DN = "%s,%s" % (_GROUP_OU, _BASE_DN)

# Authenticating with the LDAP server
AUTH_LDAP_BIND_DN = "CN=%s,%s" % (CVAT_LDAP_USER, _PEOPLE_DN)
# AUTH_LDAP_BIND_DN = "cvat_bind@ad.example.com"
AUTH_LDAP_BIND_PASSWORD = CVAT_LDAP_PASSWORD

AUTH_LDAP_USER_SEARCH = LDAPSearch(_BASE_DN, ldap.SCOPE_SUBTREE, "(uid=%(user)s)")

# AUTH_LDAP_GROUP_SEARCH = LDAPSearch(
#     "CN=Groups,%s" % _BASE_DN, ldap.SCOPE_SUBTREE, "(objectClass=groupOfNames)"
# )

AUTH_LDAP_GROUP_SEARCH = LDAPSearch(
    _GROUP_DN, ldap.SCOPE_SUBTREE, "(objectClass=groupOfUniqueNames)"
)


# Mapping Django field names to FreeIPA attributes
AUTH_LDAP_USER_ATTR_MAP = {
    "user_name": "uid",
    "first_name": "first_name",
    "last_name": "last_name",
    "email": "mail",
}

AUTH_LDAP_ALWAYS_UPDATE_USER = True

# Cache group memberships for an hour to minimize LDAP traffic
# AUTH_LDAP_CACHE_GROUPS = True
# AUTH_LDAP_GROUP_CACHE_TIMEOUT = 3600
# AUTH_LDAP_AUTHORIZE_ALL_USERS = False

# Group Management
AUTH_LDAP_GROUP_TYPE = GroupOfUniqueNamesType(name_attr="cn")
# Si GroupOfUniqueNamesType no funciona, se puede personalizar manualmente:
# class LLDAPGroupType(GroupOfNamesType):
#     member_attr = "uniquemember"  # Atributo que almacena los miembros
# AUTH_LDAP_GROUP_TYPE = LLDAPGroupType()

# AUTH_LDAP_GROUP_TYPE = GroupOfNamesType(name_attr="cn")
# AUTH_LDAP_GROUP_TYPE = GroupOfNamesType() # FreeIPA, LLDAP
# AUTH_LDAP_GROUP_TYPE = NestedActiveDirectoryGroupType() # Active Directory only

# Ajuste manual:
# AUTH_LDAP_USER_FLAGS_BY_GROUP = {
#     "is_staff": "cn=cvat_admins," + _GROUP_DN,
#     "is_superuser": "cn=cvat_admins," + _GROUP_DN,
# }
# AUTH_LDAP_FIND_GROUP_PERMS = True

# Register Django LDAP backend
AUTHENTICATION_BACKENDS += ["django_auth_ldap.backend.LDAPBackend"]

# Map FreeIPA groups to Django/CVAT groups.
AUTH_LDAP_ADMIN_GROUPS = [
    "CN=cvat_admins,%s" % _GROUP_DN,
]
AUTH_LDAP_WORKER_GROUPS = [
    "CN=cvat_workers,%s" % _GROUP_DN,
]
AUTH_LDAP_USER_GROUPS = [
    "CN=cvat_users,%s" % _GROUP_DN,
]

DJANGO_AUTH_LDAP_GROUPS = {
    "admin": AUTH_LDAP_ADMIN_GROUPS,
    "user": AUTH_LDAP_USER_GROUPS,
    "worker": AUTH_LDAP_WORKER_GROUPS,
}
