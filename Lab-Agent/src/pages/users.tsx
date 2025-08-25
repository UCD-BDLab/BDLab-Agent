import * as React from 'react';
import { Crud } from '@toolpad/core/Crud';
import { usersDataSource, User, usersCache } from '../data/users';

export default function UsersCrudPage() {
  return (
    <Crud<User>
      dataSource={usersDataSource}
      dataSourceCache={usersCache}
      rootPath="/users"
      initialPageSize={25}
      defaultValues={{ itemCount: 1 }}
    />
  );
}