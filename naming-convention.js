// Edit your custom Exclude words
const excludeWords = [
  'cloud_name',
  'api_key',
  'api_secret',
  'access_type',
  'public_id',
  'unique_filename',
  'client_email',
  'private_key',
  'project_id',
]
const regex = `${excludeWords.join('|')})$`

const defaultRule = require('eslint-config-yceffort/rules/typescript')

const extendedRules = defaultRule.rules[
  '@typescript-eslint/naming-convention'
].map((item) =>
  typeof item === 'string'
    ? item
    : {
        ...item,
        filter: {
          regex: `${item.filter.regex.split(')$')[0]}|${regex}`,
          match: false,
        },
      },
)

module.exports = {
  extendedRules,
}
