# Jekyll Website for Multi-Agent Spacecraft Docking System

This directory contains the minimalistic Jekyll website for the Multi-Agent Spacecraft Docking System project.

## Local Development

### Prerequisites

- Ruby 2.7 or higher
- Bundler gem

### Setup and Run

```bash
# Install dependencies
bundle install

# Start development server
./serve.sh

# Or manually
bundle exec jekyll serve --livereload
```

Visit `http://localhost:4000` to view the site.

## Structure

```
docs/
├── _config.yml           # Jekyll configuration
├── _layouts/             # Page templates
├── _includes/            # Reusable components
├── _sass/                # Stylesheet source
├── assets/               # Static assets
├── pages/                # Content pages
├── index.md              # Homepage
└── Gemfile               # Ruby dependencies
```

## Design Principles

- Minimalistic design with white background and black text
- No colors, emojis, or visual distractions
- Professional technical documentation focus
- Fast loading times and SEO optimized
- Responsive design for all devices

## Deployment

The site is configured for automatic deployment to GitHub Pages via GitHub Actions when changes are pushed to the main branch.

## Content Updates

To add new content:

1. Create new markdown files in the `pages/` directory
2. Add navigation links to `_includes/navigation.html`
3. Update `_config.yml` if needed for new collections or settings