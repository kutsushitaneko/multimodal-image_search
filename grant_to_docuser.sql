grant connect, ctxapp, unlimited tablespace, create credential, create procedure, create table to docuser;
grant execute on sys.dmutil_lib to docuser;
grant execute on ctxsys.ctx_ddl to docuser;
grant create mining model to docuser;
grant DWROLE to docuser;
grant create view to docuser;


BEGIN
    DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
        host => '*',
        ace => xs$ace_type(
            privilege_list => xs$name_list('connect'),
            principal_name => 'docuser',
            principal_type => xs_acl.ptype_db
        )
    );
END;
/
