---
title: '왜 Nodejs ORM을 쓰지 말아야 할까'
tags:
  - javascript
  - nodejs
  - SQL
published: true
date: 2021-07-02 19:29:54
description: 'SQL 오랫만에 보니까 반갑당'
---

## Table of Contents

Nodejs와 SQL을 사용하는 백엔드 환경을 구성하게 되면, ORM의 사용을 고려하게 된다. 그런데 왜 이 ORM의 사용을 자제해야할까?

글을 들어가기에 앞서 ORM 으로 만들어진 오픈소스를 디스(?)하기 위한 목적이 아닌 글을 밝힌다. 각 오픈소스는 열심히 만들어지고 있고, 많은 프로덕션 애플리케이션이 이를 기반으로 돌아가고 있으며, 나도 ORM으로 만들어진 백엔드 애플리케이션을 운영하고 있다.

## ORM

ORM (Object-relational mapping)은 객체와 데이터베이스 시스템을 연결(맵핑)해주는 라이브러리다. 세상엔 다양한 데이터 베이스 시스템이 있고 이를 다양한 방식으로 연결해야 하는데, ORM은 이렇게 둘 사이 (소스와 애플리케이션)의 연결을 도와주는 가교 역할을 한다. 따라서 많은 개발자들이 ORM과 데이터베이스의 마이그레이션을 편리하게 하기 위해 사용한다.

왜 쓰지 말아야 하는지에 대해 논의하기전에, 먼저 ORM의 장점을 살펴보자.

- 중복 코드 방지
- 다른 데이터베이스로 쉽게 교체 가능
- 여러 테이블에 쉽게 쿼리를 날릴 수 있음
- 인터페이스를 작성하는 시간을 아껴 비즈니스 로직에 집중할 수 있음

## ORM과 Nodejs: 추상화 계층 살펴보기

ORM에 깊게 들어가기에 앞서 추상화 계층 (abstraction layer)을 살펴보자. 컴퓨터의 다른 모든 것과 마찬가지로, 추상화 계층이 추가되는 것이 꼭 좋은 것 만은 아니다. 추가될 때 마다 성능이 저하되고, 이 성능을 발판삼아 개발자의 생산성을 향상시키게 된다 (물론 꼭 그런건 아니다)

### 저수준: 데이터베이스 드라이버

기본적으로 TCP 패킷을 수동으로 생성하여 데이터베이스를 전달하는 최소한의 과정을 제외한 가장 낮은 수준이다. 데이터베이스 드라이버 데이터베이스에 대한 연결 (풀링)을 처리한다. 이 레벨에서는 raw sql 쿼리를 작성하여 데이터베이스에 넘기고, 데이터베이스로 부터 응답을 받게된다. node.js의 생태계에서는 이러한 역할을 하는 많은 라이브러리가 있다.

- https://github.com/mysqljs/mysql
- https://github.com/brianc/node-postgres
- https://github.com/mapbox/node-sqlite3

각 라이브러리는 기본적으로 동일한 방식으로 동작한다. 데이터베이스 인증정보를 가져오고, 새 데이터베이스 인스턴스를 만들고, 연결하고, 문자열 형식으로 쿼리를 전송하고, 결과를 비동기적으로 처리한다.

```javascript
#!/usr/bin/env node

// $ npm install pg

const {Client} = require('pg')
const connection = require('./connection.json')
const client = new Client(connection)

client.connect()

const query = `SELECT
  ingredient.*, item.name AS item_name, item.type AS item_type
FROM
  ingredient
LEFT JOIN
  item ON item.id = ingredient.item_id
WHERE
  ingredient.dish_id = $1`

client.query(query, [1]).then((res) => {
  console.log('Ingredients:')
  for (let row of res.rows) {
    console.log(`${row.item_name}: ${row.quantity} ${row.unit}`)
  }

  client.end()
})
```

### 중간수준: 쿼리 빌더

이는 데이터베이스 드라이버 모듈을 사용하는 것과 완전한 ORM을 사용하는 것 사이의 중간 정도 수준이다. 이 정도 수준을 구현한 라이브러리는 [Knext](https://knexjs.org/)다. 이 라이브러리는 여러가지 다른 SQL 구문을 만들어낼 수 있다. Knex는 함께 사용해야 하는 다른 라이브러리를 같이 설치해줘야 한다.

Knex 인스턴스 생성시 사용하는 SQL과 함께 연결관련 정보를 넘겨서 사용할 수 있다. 작성하는 쿼리는 기본적으로 SQL 쿼리와 유사하다. 한가지 좋은 점은 문자열을 연결해서 SQL 쿼리를 만드는 것 보다는 더 편리한 방식으로 동적 쿼리를 프로그래밍 방식으로 생성할 수 있다는 것이다. (가끔 보안 취약점이 발생할 수도 있음)

```javascript
#!/usr/bin/env node

// $ npm install pg knex

const knex = require('knex')
const connection = require('./connection.json')
const client = knex({
  client: 'pg',
  connection,
})

client
  .select([
    '*',
    client.ref('item.name').as('item_name'),
    client.ref('item.type').as('item_type'),
  ])
  .from('ingredient')
  .leftJoin('item', 'item.id', 'ingredient.item_id')
  .where('dish_id', '=', 1)
  .debug()
  .then((rows) => {
    console.log('Ingredients:')
    for (let row of rows) {
      console.log(`${row.item_name}: ${row.quantity} ${row.unit}`)
    }

    client.destroy()
  })
```

### 고수준: ORM

이제 우리가 고려할 가장 높은 수준의 추상화다. 써본 사람들은 알겠지만, 일반적으로 ORM을 사용할 때는 사전에 설정을 하는데 많은 시간을 쏟아야 한다. ORM의 요점은, 이름에서 알 수 있는 것처럼 관계형 데이터 베이스의 데이터를 애플리케이션의 객체 (일반적으로 클래스 인스턴스)에 매핑하는 것이다. 따라서 애플리케이션 코드에서 이러한 객체의 구조와 관계를 정의해야 한다.

- https://github.com/sequelize/sequelize
- https://github.com/bookshelf/bookshelf
- https://github.com/balderdashy/waterline
- https://github.com/Vincit/objection.js

## Sequelize 사용해보기

가장 유명한 ORM인 `Sequelize`를 사용해보자.

```javascript
#!/usr/bin/env node

// $ npm install sequelize pg

const Sequelize = require('sequelize')
const connection = require('./connection.json')
const DISABLE_SEQUELIZE_DEFAULTS = {
  timestamps: false,
  freezeTableName: true,
}

const {DataTypes} = Sequelize
const sequelize = new Sequelize({
  database: connection.database,
  username: connection.user,
  host: connection.host,
  port: connection.port,
  password: connection.password,
  dialect: 'postgres',
  operatorsAliases: false,
})

const Dish = sequelize.define(
  'dish',
  {
    id: {type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true},
    name: {type: DataTypes.STRING},
    veg: {type: DataTypes.BOOLEAN},
  },
  DISABLE_SEQUELIZE_DEFAULTS,
)

const Item = sequelize.define(
  'item',
  {
    id: {type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true},
    name: {type: DataTypes.STRING},
    type: {type: DataTypes.STRING},
  },
  DISABLE_SEQUELIZE_DEFAULTS,
)

const Ingredient = sequelize.define(
  'ingredient',
  {
    dish_id: {type: DataTypes.INTEGER, primaryKey: true},
    item_id: {type: DataTypes.INTEGER, primaryKey: true},
    quantity: {type: DataTypes.FLOAT},
    unit: {type: DataTypes.STRING},
  },
  DISABLE_SEQUELIZE_DEFAULTS,
)

Item.belongsToMany(Dish, {
  through: Ingredient,
  foreignKey: 'item_id',
})

Dish.belongsToMany(Item, {
  through: Ingredient,
  foreignKey: 'dish_id',
})

Dish.findOne({where: {id: 1}, include: [{model: Item}]}).then((rows) => {
  console.log('Ingredients:')
  for (let row of rows.items) {
    console.log(
      `${row.dataValues.name}: ${row.ingredient.dataValues.quantity} ` +
        row.ingredient.dataValues.unit,
    )
  }

  sequelize.close()
})
```

## 정말 ORM이 필요한가?

### 1. SQL이 아닌 ORM 자체를 배우게 된다.

많은 사람들이 SQL을 배우기 위해 ORM을 선택한다. 사람들은 SQL은 배우기 어렵고, ORM을 배우면 하나의 언어만 사용하여 애플리케이션을 작성할 수 있다는 믿음을 갖곤 한다. 언뜻 보기에 이는 맞는 것 같다. 그러나 ORM은 애플리케이션의 나머지 언어와 동일하게 작성되지만, SQL은 완전히 다른 문법을 가진 언어다.

이러한 사고방식에는 문제가 있다. ORM은 꽤 복잡한 라이브러리중 하나다. ORM을 사용하기 위해서 배워야할 것은 많으며, 이를 배우는 것은 결코 쉬운일이 아니다.

그리고 특정 ORM에 익숙해져버리면, 다른 ORM 사용을 원활하게 하지 못할 수도 있다. 이는 마치 JS/Node.js에서 C#/.NET 환경으로 오는 것과 유사한 기분이 들 수 있다.

#### Sequelize

```javascript
#!/usr/bin/env node

// $ npm install sequelize pg

const Sequelize = require('sequelize')
const {Op, DataTypes} = Sequelize
const connection = require('./connection.json')
const DISABLE_SEQUELIZE_DEFAULTS = {
  timestamps: false,
  freezeTableName: true,
}

const sequelize = new Sequelize({
  database: connection.database,
  username: connection.user,
  host: connection.host,
  port: connection.port,
  password: connection.password,
  dialect: 'postgres',
  operatorsAliases: false,
})

const Item = sequelize.define(
  'item',
  {
    id: {type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true},
    name: {type: DataTypes.STRING},
    type: {type: DataTypes.STRING},
  },
  DISABLE_SEQUELIZE_DEFAULTS,
)

// SELECT "id", "name", "type" FROM "item" AS "item"
//     WHERE "item"."type" = 'veg';
Item.findAll({where: {type: 'veg'}}).then((rows) => {
  console.log('Veggies:')
  for (let row of rows) {
    console.log(`${row.dataValues.id}t${row.dataValues.name}`)
  }
  sequelize.close()
})
```

#### Bookshelf

```javascript
#!/usr/bin/env node

// $ npm install bookshelf knex pg

const connection = require('./connection.json')
const knex = require('knex')({
  client: 'pg',
  connection,
  // debug: true
})
const bookshelf = require('bookshelf')(knex)

const Item = bookshelf.Model.extend({
  tableName: 'item',
})

// select "item".* from "item" where "type" = ?
Item.where('type', 'veg')
  .fetchAll()
  .then((result) => {
    console.log('Veggies:')
    for (let row of result.models) {
      console.log(`${row.attributes.id}t${row.attributes.name}`)
    }
    knex.destroy()
  })
```

#### Waterline

```javascript
#!/usr/bin/env node

// $ npm install sails-postgresql waterline

const pgAdapter = require('sails-postgresql')
const Waterline = require('waterline')
const waterline = new Waterline()
const connection = require('./connection.json')

const itemCollection = Waterline.Collection.extend({
  identity: 'item',
  datastore: 'default',
  primaryKey: 'id',
  attributes: {
    id: {type: 'number', autoMigrations: {autoIncrement: true}},
    name: {type: 'string', required: true},
    type: {type: 'string', required: true},
  },
})

waterline.registerModel(itemCollection)

const config = {
  adapters: {
    pg: pgAdapter,
  },

  datastores: {
    default: {
      adapter: 'pg',
      host: connection.host,
      port: connection.port,
      database: connection.database,
      user: connection.user,
      password: connection.password,
    },
  },
}

waterline.initialize(config, (err, ontology) => {
  const Item = ontology.collections.item
  // select "id", "name", "type" from "public"."item"
  //     where "type" = $1 limit 9007199254740991
  Item.find({type: 'veg'}).then((rows) => {
    console.log('Veggies:')
    for (let row of rows) {
      console.log(`${row.id}t${row.name}`)
    }
    Waterline.stop(waterline, () => {})
  })
})
```

#### Objection

```javascript
#!/usr/bin/env node

// $ npm install knex objection pg

const connection = require('./connection.json')
const knex = require('knex')({
  client: 'pg',
  connection,
  // debug: true
})
const {Model} = require('objection')

Model.knex(knex)

class Item extends Model {
  static get tableName() {
    return 'item'
  }
}

// select "item".* from "item" where "type" = ?
Item.query()
  .where('type', '=', 'veg')
  .then((rows) => {
    for (let row of rows) {
      console.log(`${row.id}t${row.name}`)
    }
    knex.destroy()
  })
```

단순한 읽기 작업이지만, 예제 사이에 많은 차이가 존재하는 것을 볼 수있다. 이는 여러 테이블을 조인하게 되면 작업이 복잡해지면서 ORM 구문이 더 복잡하고 달라질 수 있다. Node.js에는 수많은 ORM이 있고, 또 모든 플랫폼에 대해 수백개의 ORM이 존재한다. 이를 다 배운다는 것은 악몽과도 같다.

그러나 SQL 구문을 배운다면 이런걱정을 할 필요가 없다. SQL을 사용하여 쿼리를 생성하는 방법을 배우게 되면, 이 하나의 지식을 여러 플랫폼 사이에서 공유할 수 있다.

### 2. 복잡한 ORM 호출은 비효율적일 수 있다.

ORM의 본래 목적은 데이터베이스에 저장된 기본 데이터를, 특정 애플리케이션 내에서 상호작용할 수 있는 객체에 매핑하는 것이다. 따라서 ORM을 사용하여 데이터를 가져올 때 몇가지 비효율성이 발생하게 된다.

아래 추상화 레벨별 예제를 살펴보자.

#### `pg` 드라이버 사용

이 방법에서는 쿼리를 직접 손으로 쓰면 된다. 이는 우리가 원하는 데이터를 얻을 수 있는 가장 간결한 방법이다.

```sql
SELECT
  ingredient.*, item.name AS item_name, item.type AS item_type
FROM
  ingredient
LEFT JOIN
  item ON item.id = ingredient.item_id
WHERE
ingredient.dish_id = ?;
```

이 쿼리를 `EXPLAIN`으로 살펴보면, `34.12`의 비용이 나온다.

#### `knex` 쿼리 빌더 사용시

```sql
select
  *, "item"."name" as "item_name", "item"."type" as "item_type"
from
  "ingredient"
left join
  "item" on "item"."id" = "ingredient"."item_id"
where
"dish_id" = ?;
```

앞선 예와 마찬가지로 일부 사소한 형식 `"`과 불필요한 몇가지를 제외하면 동일하다. 마찬가지의 비용인 `34.12`가 나온다.

#### Sequelize ORM

이제 ORM으로 생성한 쿼리를 살펴보자.

```sql
SELECT
  "dish"."id", "dish"."name", "dish"."veg", "items"."id" AS "items.id",
  "items"."name" AS "items.name", "items"."type" AS "items.type",
  "items->ingredient"."dish_id" AS "items.ingredient.dish_id",
  "items->ingredient"."item_id" AS "items.ingredient.item_id",
  "items->ingredient"."quantity" AS "items.ingredient.quantity",
  "items->ingredient"."unit" AS "items.ingredient.unit"
FROM
  "dish" AS "dish"
LEFT OUTER JOIN (
  "ingredient" AS "items->ingredient"
  INNER JOIN
  "item" AS "items" ON "items"."id" = "items->ingredient"."item_id"
) ON "dish"."id" = "items->ingredient"."dish_id"
WHERE
"dish"."id" = ?;
```

이 쿼리는 앞선 쿼리와는 많이 다르다. 앞서 정의한 `관계` 때문에, Sequelize는 요청한 것보다 더 많은 정보를 얻으려고 한다. 이 쿼리의 비용은 `42.32`다.

### 3. ORM이 만능은 아니다.

일부 쿼리는 ORM 작업으로 표현할 수 없다. 이러한 쿼리를 생성하는 경우에는 SQL 쿼리를 직접생성하는 작업으로 회귀해야 한다. 이는 ORM을 사용하는 와중에도 코드베이스에 여전히 하드 코딩된 쿼리가 존재할 수 있다는 것을 의미한다. 이러한 프로젝트를 개발하는 개발자는 ORM이나 SQL구문 모두를 알아야 한다.

ORM으로 표현할 수 없는 쿼리에는 쿼리에 서브쿼리가 포함된 경우다.

```sql
SELECT *
FROM item
WHERE
  id NOT IN
    (SELECT item_id FROM ingredient WHERE dish_id = 2)
  AND id IN
(SELECT item_id FROM ingredient WHERE dish_id = 1);
```

내가 아는한, 이 쿼리는 앞서 언급한 ORM을 사용하여 명확하게 나타낼 수 없다. 이러한 상황에 대처하기 위해, ORM에서는 쿼리인터페이스에 로우 쿼리 문자열을 주입하는 기능을 제공하는 것이 일반적이다.

Sequelize의 경우에는 로우 쿼리문을 실행하는 `.query()`메소드를 제공한다. Bookshelf와 Objection ORM을 사용하면, 로우 knex 객체에 엑세스할 수 있다. Knex 객채에는 로우 쿼리를 실행하는 `.raw()` 메소드도 있다. 어쨌건 간에, 여전히 특정 쿼리를 사용하기 위해서는 SQL을 이해해야한다.

## 쿼리 빌더를 사용하자.

저수준 데이터베이스 드라이버 모듈을 사용하는 것은 매력적이다. 쿼리를 손수 작성하므로, 데이터베이스에 쿼리를 생성할 때 오버헤드를 일으키지 않는다. 또한 프로젝트에 의존하는 전반적인 의존성도 최소화 할 수 있다. 그러나 동적쿼리를 생성하는 작업이 매우 귀찮을 수 있다.

사용자가 특정 기준에 따라 데이터를 가져오는 예제를 상상해보자.

```sql
SELECT * FROM things WHERE color = ?;
```

그러나 옵션이 다양해지면 아래와 같이 복잡해진다.

```sql
SELECT * FROM things; -- Neither
SELECT * FROM things WHERE color = ?; -- Color only
SELECT * FROM things WHERE is_heavy = ?; -- Is Heavy only
SELECT * FROM things WHERE color = ? AND is_heavy = ?; -- Both
```

매우 복잡해졌다. 그러나 앞선 이유로 인해 우리는 ORM을 사용하지 않을 것이다. 그렇다면?

쿼리 빌더가 매우 유용하게 쓰일 수 있다. knex에서 제공하는 인터페이스는 기본 SQL 쿼리와 매우 유사하므로, SQL 쿼리에 대해서 잘 이해해야 한다. 이는 typescript가 javascript로 변환되는 것과 유사하다.

SQL에 대해서 완전히 이해했다면, 쿼리빌더를 사용하는 것이 가장 좋은 해결책이다. 절대로 하위 계층에서 발생하는 일을 외면하기 위한 도구로 사용하지 말자. 편의상의 이유로만 사용해야 하며, 반드시 무슨일이 벌어지고 있는지 이해해야 한다. `Knex()`에 debug옵션을 추가하면 쿼리가 어떻게 생성되는지 알 수 있다.

```javascript
const knex = require('knex')({
  client: 'pg',
  connection,
  debug: true, // Enable Query Debugging
})
```

> ORM의 존재에 대한 갑론을박이 이렇게 뜨거운줄은 몰랐다. 엔터프아리즈급 백엔드를 많이 작성해본 경험이 없어서 이게 정답이다라고 뚜렷하게 이야기할 수는 없지만, 이런저런 측면에서 고민해볼만한 주제인 것 같긴 하다. 그러나 한가지 공통된 의견은, SQL을 꼭 배워야 한다는 것. 프론드엔드 개발자 또한 마찬가지로.

- https://blog.logrocket.com/why-you-should-avoid-orms-with-examples-in-node-js-e0baab73fa5/comment-page-1/#comments
- https://martinfowler.com/bliki/OrmHate.html
- https://enterprisecraftsmanship.com/posts/do-you-need-an-orm/
- https://medium.com/@mithunsasidharan/should-i-or-should-i-not-use-orm-4c3742a639ce
- https://hackernoon.com/you-dont-need-an-orm-7ef83bd1b37d
