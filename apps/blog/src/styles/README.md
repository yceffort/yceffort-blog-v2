# Blog styles

The blog uses StyleX for component UI and CSS for document and generated-content selectors. The research app uses Tailwind.

## Component styles

Keep short definitions in the component and larger definitions in an adjacent `*.styles.ts`. Put the shared `@layer` around the properties once. A style applied directly to a component does not need to repeat that component's class selector:

```ts
const styles = stylex.create({
  hero: {
    '@layer site': {
      position: 'relative',
      padding: {
        default: '32px 0 28px',
        '@media (max-width: 640px)': '20px 0 16px',
      },
    },
  },
})
```

Keep media, support, and state conditions inside the affected property's value. Separate sibling conditions can exclude one another when StyleX compiles them. Use consistent border longhands where state styles override only some border properties.

A condition key must describe the element it is attached to, never an ancestor. `':hover'`, `':last-child'` and `':is([data-open="true"])'` are fine; `':is(.post-card:hover .thumb img)'` is not, because the wrapping selector reintroduces the CSS-file cascade that owning styles per element is meant to replace, and it makes the compiler pad every rule with extra specificity.

The one ancestor allowed is the document theme marker: `':is(.dark *)'` (and `':is(.dark *):hover'`). `next-themes` sets `.dark` on `<html>`, no element can own that state, and the alternative of a custom property per themed value would duplicate every color token. Keep it to theme colors; do not use it for layout.

Three rules cover what a descendant selector used to do:

- **A parent state changes a child.** The parent sets a custom property under its own state; the child reads it with the value it had before as the fallback. `post_card` sets `'--post-card-thumb-transform'` under `':hover'` and the image uses `transform: 'var(--post-card-thumb-transform, none)'`.
- **A shared style needs different values per parent.** Split it into one entry per parent, export both, and let each consumer pick. `list.styles.ts` keeps the common `series` entry and adds `series_in_post_row` and `series_in_post_card`.
- **React owns the state.** Select the style in the component instead of reading the attribute in CSS, the way the header picks the active navigation label.

Style an element with a single `stylex.props` call. When two entries both define a property, compose them (`stylex.props(styles.post_row_thumb, styles.post_row_thumb_empty)`) so StyleX resolves the winner; two class strings joined in JSX leave it to stylesheet order.

Compose variants with `stylex.props` in their defining module and export the resulting class constants. For example, `home/recent.styles.ts` composes the series-row variant locally; `RecentRow` selects it with a `variant` prop. Do not concatenate competing style definitions and rely on class-string order to choose a winner.

Shared accessibility styles belong in `styles/accessibility.styles.ts`. A component must not import About or another route just to hide a label or render an external link. Shared MobileNav and SocialIcon receive classes through their app adapters.

## CSS boundaries

`stylex.css` is the global CSS entry and `reading.css` the long-form one. Their imports have these owners:

- `browser.css`: reset, non-inheriting custom-property registrations, selection, and the theme-fade fallback.
- `base.css` and `elements.css`: document element defaults and anchor behavior.
- `theme.css`: document colors and fonts; next-themes supplies the `dark` marker.
- `transitions.css`: navigation and theme View Transitions.
- `spacing.css`: the flow spacing Tailwind's `space-*` utilities used to provide for layout wrappers outside Markdown.
- `reading.css`: the long-form entry. It is imported by the components that render Markdown, not by `stylex.css`, so list, tag and archive routes never download it. Its members are `typography.css`, `post/article.css`, `post/markdown.css` and `post/markdown-toc.css`.
- `typography.css`: Markdown typography and contextual spacing. Nested content can opt out with `markdown-exempt`.
- `post/article.css`: article and series-content selectors, including external-link indicators.
- `post/markdown.css` and `post/markdown-toc.css`: code/diagram output and generated tables of contents.
- `settings/appearance.css`: user-selected palettes and effects. The `preferences` layer follows component styles so minimal mode can disable their animations.

A rule that only long-form pages need belongs in `reading.css`; anything the header, footer or list routes rely on belongs in the global entry. `typography.css` held two Tailwind `space-*` leftovers that the footer and the 404 page depended on, and moving the file without them broke every list route.

`tokens.stylex.ts` defines the shared CSS-variable names used by both authoring formats. Semantic markers such as `post-article`, `series-thread`, and `markdown-exempt` remain where CSS or DOM behavior needs them.

Markdown styling does not require a rehype traversal or generated classes on every heading, paragraph, link, Prism token, or KaTeX node. MDX wrappers preserve incoming properties and add only the styles they own. WASM produces the code token classes, line metadata, MathML and image dimensions. `post/markdown.css` colors those token classes and supplies the local math font; `parseCodeSnippet` is no longer part of the render path.

## Compilation and verification

The Babel and PostCSS integrations share compiler settings. Runtime injection is disabled. Local CSS imports are processed in one pass so production optimization preserves the cascade and color precision across files. The production optimizer's compatibility conversion is retained for visual parity.

Run `pnpm --filter blog test:styles` for the Markdown CSS checks (code-block typography plus Prism token colors). They ensure inline-code rules cannot shrink or bold block code across widths, themes, and article/typography contexts.

Build and start the baseline and the current app on separate ports, then run:

```sh
pnpm --filter blog test:styles:pages http://localhost:3104 http://localhost:3100
pnpm --filter blog test:styles:motion http://localhost:3104 http://localhost:3100
```

`STYLE_PARITY_ROUTES` limits the comparison to selected routes. `/about` and `/resume` changed content in `7c4fa58e`, so compare those two against a build of that commit rather than against `main`.

Page comparisons cover 13 routes at desktop/mobile widths in light/dark themes, plus menu, search, settings, palettes, film grain, and minimal mode. They compare computed properties, pseudo-elements, content, geometry, and screenshots with zero tolerance. Motion checks compare animation timing and keyframes separately, including minimal mode.

Snapshots freeze animations and mask WebGL canvas pixels; they do not claim identical JavaScript/WebGL animation frames. Existing screenshot normalization for idle tag compositing remains. Preserve JSX text-node boundaries: changing whitespace nodes can move glyphs even when CSS is identical.

`STYLE_PARITY_ROUTES`, `STYLE_PARITY_WIDTHS`, `STYLE_PARITY_THEMES`, `STYLE_PARITY_STATES=0`, and `STYLE_PARITY_DIR` support focused page comparisons. Reports are written under `.cache/style-parity/`. Historical migration/performance reports in `tests/` describe their recorded revisions, not automatic guarantees for later changes.

See the [refactor verification report](../../tests/styles/REFACTOR.md) for the preserved baseline, measured sizes, and results of these checks.
