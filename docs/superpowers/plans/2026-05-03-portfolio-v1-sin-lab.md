# Portfolio v1 (sin /lab) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar el portfolio HTML actual por un sitio Astro estático en producción en `emanuelancco.com` con 12 proyectos validados, /archive, /cv, /sobre, /contacto. Sin /lab (va en plan separado).

**Architecture:** Astro 5 SSG + Tailwind 4 + Content Collections tipados con Zod. Deploy en Cloudflare Pages con custom domain. Imágenes optimizadas vía `astro:assets`. Cero JS por defecto en home (islas solo en /lab futuro).

**Tech Stack:** Astro 5 · Tailwind CSS 4 · TypeScript strict · Cloudflare Pages · Cloudflare API REST · Fraunces + Inter + JetBrains Mono (self-hosted) · pnpm.

**Spec de referencia:** [`docs/superpowers/specs/2026-05-03-portfolio-emanuel-design.md`](../specs/2026-05-03-portfolio-emanuel-design.md)

**Convención de commits:** sin firma de Claude, sin Co-Authored-By, formato Conventional Commits.

**Pre-requisito humano (1 vez):** generar API token de Cloudflare con scopes `Account.Cloudflare Pages: Edit`, `Account.Account Settings: Read`, `Zone.DNS: Edit`, `Zone.Zone: Read`. Guardar en `.env.local` (gitignored) como `CLOUDFLARE_API_TOKEN=...` y `CLOUDFLARE_ACCOUNT_ID=...`.

---

## Stage A — Scaffold y CI

### Task A1: Backup del HTML actual antes de tocar nada

**Files:**
- Create: `legacy/index.html.2026-05-03.bak`

- [ ] **Step 1: Verificar working tree limpio (excepto archivos no rastreados conocidos)**

```bash
git status -s
```

Expected: solo aparecen `.superpowers/`, `CV_GenPlus_IA_2026.tex`, `CV_Las_Bambas_Geotecnia_2026.tex`. Si hay otros cambios sin commit, parar y preguntar.

- [ ] **Step 2: Crear copia del HTML actual en `legacy/`**

```bash
mkdir -p legacy
cp index.html legacy/index.html.2026-05-03.bak
```

- [ ] **Step 3: Commit**

```bash
git add legacy/
git commit -m "chore: backup HTML portfolio v1 before Astro migration"
```

---

### Task A2: Inicializar proyecto Astro en `site/` (subcarpeta del repo)

**Files:**
- Create: `site/package.json`, `site/astro.config.mjs`, `site/tsconfig.json`, `site/src/`, `site/public/`

Razón de subcarpeta: el repo ya tiene `index.html`, `codigosrealizados/`, `assets/`, CVs `.tex` y `legacy/`. Mantener Astro en `site/` aísla y no rompe nada.

- [ ] **Step 1: Verificar Node 22+ y pnpm instalados**

```bash
node -v
pnpm -v
```

Expected: Node `v22.x.x` o superior, pnpm `8.x` o superior. Si pnpm falta: `npm i -g pnpm`.

- [ ] **Step 2: Inicializar Astro mínimo en `site/`**

```bash
pnpm create astro@latest site -- --template minimal --typescript strict --install --no-git
```

Expected: Astro pregunta y se queda con minimal + strict TS + instala deps. Carpeta `site/` poblada.

- [ ] **Step 3: Verificar dev server arranca**

```bash
cd site && pnpm dev
```

Expected: server en `http://localhost:4321`, página "Astro" por defecto. `Ctrl+C` para parar.

- [ ] **Step 4: Commit**

```bash
git add site/
git commit -m "feat(site): scaffold Astro project in site/ subfolder"
```

---

### Task A3: Añadir Tailwind CSS 4 vía Vite plugin (no integration legacy)

**Files:**
- Modify: `site/astro.config.mjs`
- Modify: `site/package.json`
- Create: `site/src/styles/globals.css`

- [ ] **Step 1: Instalar Tailwind 4 + Vite plugin**

```bash
cd site && pnpm add -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: Reemplazar `site/astro.config.mjs` con esto**

```js
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://emanuelancco.com',
  vite: {
    plugins: [tailwindcss()],
  },
});
```

- [ ] **Step 3: Crear `site/src/styles/globals.css`**

```css
@import "tailwindcss";

@theme {
  --color-bg: #020617;
  --color-surface: #0f172a;
  --color-text: #f8fafc;
  --color-text-dim: #94a3b8;
  --color-accent: #f59e0b;
  --color-accent-light: #fbbf24;
  --color-rule: rgba(255, 255, 255, 0.08);

  --font-display: "Fraunces Variable", Georgia, serif;
  --font-body: "Inter Variable", system-ui, sans-serif;
  --font-mono: "JetBrains Mono Variable", ui-monospace, monospace;

  --container-1100: 1100px;
}

html { background: var(--color-bg); color: var(--color-text); }
body { font-family: var(--font-body); }
```

- [ ] **Step 4: Verificar dev arranca sin errores**

```bash
cd site && pnpm dev
```

Expected: sin warnings de Tailwind, página por defecto sigue cargando. `Ctrl+C`.

- [ ] **Step 5: Commit**

```bash
git add site/
git commit -m "feat(site): add Tailwind 4 with custom theme tokens"
```

---

### Task A4: Self-host fuentes (Fraunces, Inter, JetBrains Mono) variables

**Files:**
- Create: `site/public/fonts/Fraunces[opsz,SOFT,WONK,wght].woff2`
- Create: `site/public/fonts/InterVariable.woff2`
- Create: `site/public/fonts/JetBrainsMono[wght].woff2`
- Modify: `site/src/styles/globals.css`

- [ ] **Step 1: Descargar las 3 fuentes variables (woff2) desde fontsource o GitHub oficial**

```bash
mkdir -p site/public/fonts
cd site/public/fonts

# Fraunces
curl -L -o "Fraunces.woff2" "https://cdn.jsdelivr.net/fontsource/fonts/fraunces:vf@latest/latin-wght-normal.woff2"
# Inter
curl -L -o "InterVariable.woff2" "https://cdn.jsdelivr.net/fontsource/fonts/inter:vf@latest/latin-wght-normal.woff2"
# JetBrains Mono
curl -L -o "JetBrainsMono.woff2" "https://cdn.jsdelivr.net/fontsource/fonts/jetbrains-mono:vf@latest/latin-wght-normal.woff2"

ls -lh
```

Expected: 3 archivos `.woff2`, cada uno entre 30-150 KB.

- [ ] **Step 2: Añadir `@font-face` en `site/src/styles/globals.css` (al inicio, antes de `@import`)**

```css
@font-face {
  font-family: "Fraunces Variable";
  src: url("/fonts/Fraunces.woff2") format("woff2-variations");
  font-weight: 100 900;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: "Inter Variable";
  src: url("/fonts/InterVariable.woff2") format("woff2-variations");
  font-weight: 100 900;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: "JetBrains Mono Variable";
  src: url("/fonts/JetBrainsMono.woff2") format("woff2-variations");
  font-weight: 100 800;
  font-style: normal;
  font-display: swap;
}

@import "tailwindcss";

/* (resto del archivo ya existente) */
```

- [ ] **Step 3: Verificar que `globals.css` se importa desde el layout (lo crearemos en Task A5, anota que falta)**

No hay paso ejecutable aquí — pasamos a A5.

- [ ] **Step 4: Commit**

```bash
git add site/public/fonts site/src/styles/globals.css
git commit -m "feat(site): self-host Fraunces, Inter, JetBrains Mono variable fonts"
```

---

### Task A5: Layout base `Base.astro` con head, fonts preload, footer

**Files:**
- Create: `site/src/layouts/Base.astro`
- Create: `site/src/components/Footer.astro`

- [ ] **Step 1: Crear `site/src/components/Footer.astro`**

```astro
---
const year = new Date().getFullYear();
---
<footer class="border-t border-[var(--color-rule)] mt-32 py-12 text-sm text-[var(--color-text-dim)]">
  <div class="max-w-[var(--container-1100)] mx-auto px-6 flex flex-col md:flex-row justify-between gap-4">
    <div>© {year} Emanuel Ancco · Construido en Astro · Lima / Puno</div>
    <div class="flex gap-6">
      <a href="https://github.com/EmanuelAncco" class="hover:text-[var(--color-accent)]">GitHub</a>
      <a href="/contacto" class="hover:text-[var(--color-accent)]">Contacto</a>
    </div>
  </div>
</footer>
```

- [ ] **Step 2: Crear `site/src/layouts/Base.astro`**

```astro
---
import "../styles/globals.css";
import Footer from "../components/Footer.astro";

interface Props {
  title: string;
  description?: string;
  ogImage?: string;
}

const { title, description = "Portfolio de Emanuel Ancco — Ingeniería civil, IA, hardware y emprendimiento desde el sur del Perú.", ogImage = "/og-default.jpg" } = Astro.props;
const canonical = new URL(Astro.url.pathname, Astro.site).toString();
---
<!doctype html>
<html lang="es">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="canonical" href={canonical} />
    <link rel="preload" href="/fonts/Fraunces.woff2" as="font" type="font/woff2" crossorigin />
    <link rel="preload" href="/fonts/InterVariable.woff2" as="font" type="font/woff2" crossorigin />
    <title>{title}</title>
    <meta name="description" content={description} />
    <meta property="og:title" content={title} />
    <meta property="og:description" content={description} />
    <meta property="og:image" content={new URL(ogImage, Astro.site).toString()} />
    <meta property="og:url" content={canonical} />
    <meta property="og:type" content="website" />
    <meta name="twitter:card" content="summary_large_image" />
  </head>
  <body class="min-h-screen flex flex-col">
    <main class="flex-1"><slot /></main>
    <Footer />
  </body>
</html>
```

- [ ] **Step 3: Reemplazar `site/src/pages/index.astro` con uso del Base**

```astro
---
import Base from "../layouts/Base.astro";
---
<Base title="Emanuel Ancco — Portfolio">
  <section class="max-w-[var(--container-1100)] mx-auto px-6 pt-32">
    <p class="text-sm tracking-widest uppercase text-[var(--color-text-dim)]">scaffold ok</p>
    <h1 class="font-[var(--font-display)] text-6xl mt-4">Hola.</h1>
  </section>
</Base>
```

- [ ] **Step 4: Verificar render en dev**

```bash
cd site && pnpm dev
```

Expected: en `http://localhost:4321` ves "scaffold ok" en pequeño + "Hola." grande con serif Fraunces, footer abajo. DevTools Network: las 3 fuentes cargan desde `/fonts/`. `Ctrl+C`.

- [ ] **Step 5: Commit**

```bash
git add site/
git commit -m "feat(site): add Base layout, Footer component, font preload"
```

---

### Task A6: Configurar `.gitignore` y `.env.local` para credenciales Cloudflare

**Files:**
- Modify: `.gitignore` (raíz del repo)
- Create: `.env.local.example`

- [ ] **Step 1: Crear/actualizar `.gitignore` en raíz del repo**

```bash
cat >> .gitignore <<'EOF'

# Astro / Node
site/node_modules/
site/dist/
site/.astro/
site/.cache/

# Env
.env.local
.env.*.local

# Brainstorm artifacts
.superpowers/
EOF
```

- [ ] **Step 2: Crear `.env.local.example` (commiteable, sin valores)**

```bash
cat > .env.local.example <<'EOF'
# Cloudflare API token con scopes: Pages:Edit, Account:Read, DNS:Edit (zone), Zone:Read
CLOUDFLARE_API_TOKEN=
CLOUDFLARE_ACCOUNT_ID=
# Opcional: zone id de emanuelancco.com (lo descubrimos vía API si falta)
CLOUDFLARE_ZONE_ID=
EOF
```

- [ ] **Step 3: HUMANO — generar el token y rellenar `.env.local`**

Acción del usuario (no ejecutable por agent):
1. Ir a https://dash.cloudflare.com/profile/api-tokens
2. "Create Token" → "Custom token"
3. Permisos: `Account · Cloudflare Pages · Edit`, `Account · Account Settings · Read`, `Zone · DNS · Edit`, `Zone · Zone · Read`
4. Account Resources: incluir solo la cuenta donde compraste el dominio
5. Zone Resources: `emanuelancco.com`
6. Crear, copiar el token (solo se muestra 1 vez)
7. `cp .env.local.example .env.local` y pegar el token + ACCOUNT_ID (visible en sidebar de cualquier página de Cloudflare Dashboard)

- [ ] **Step 4: Verificar que `.env.local` no se va a commit**

```bash
git status -s | grep -i env
```

Expected: NO aparece `.env.local`. Solo aparece `.env.local.example`.

- [ ] **Step 5: Commit del example y .gitignore**

```bash
git add .gitignore .env.local.example
git commit -m "chore: add .env.local.example and gitignore for Astro/secrets"
```

---

### Task A7: Crear proyecto Cloudflare Pages vía API + conectar repo GitHub

**Files:** ninguno (operación remota vía API)

- [ ] **Step 1: Cargar variables y verificar API token funciona**

```bash
set -a; source .env.local; set +a
curl -sS -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/user/tokens/verify" | jq .
```

Expected: `{"result":{"status":"active"},"success":true,...}`. Si `success:false`, regenerar token.

- [ ] **Step 2: HUMANO — autorizar GitHub en Cloudflare (1 vez)**

La conexión inicial GitHub ↔ Cloudflare requiere OAuth desde el navegador, no se automatiza. Acción manual:
1. Dashboard Cloudflare → Workers & Pages → Create → Pages → Connect to Git
2. Click "Connect GitHub" → autorizar Cloudflare en el repo `EmanuelAncco/cv-portfolio`
3. Volver atrás (no crear el proyecto vía UI, lo creamos por API)

- [ ] **Step 3: Crear el proyecto Pages vía API**

```bash
set -a; source .env.local; set +a
curl -sS -X POST \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects" \
  -d '{
    "name": "emanuelancco",
    "production_branch": "main",
    "build_config": {
      "build_command": "pnpm install && pnpm build",
      "destination_dir": "dist",
      "root_dir": "site"
    },
    "deployment_configs": {
      "production": {
        "env_vars": {
          "NODE_VERSION": {"value": "22"},
          "PNPM_VERSION": {"value": "9"}
        }
      },
      "preview": {
        "env_vars": {
          "NODE_VERSION": {"value": "22"},
          "PNPM_VERSION": {"value": "9"}
        }
      }
    },
    "source": {
      "type": "github",
      "config": {
        "owner": "EmanuelAncco",
        "repo_name": "cv-portfolio",
        "production_branch": "main",
        "pr_comments_enabled": true,
        "deployments_enabled": true,
        "production_deployment_enabled": true
      }
    }
  }' | jq .
```

Expected: `{"result":{"name":"emanuelancco","subdomain":"emanuelancco.pages.dev",...},"success":true}`. Si falla con 400 sobre source, ir a UI de Cloudflare → Pages → Create → seleccionar el repo manualmente (la primera vinculación a veces falla por API). Importante el `name` en minúsculas y sin guiones.

- [ ] **Step 4: Push de la rama main para disparar primer build**

```bash
cd /c/Users/Emanuel/.gemini/antigravity/scratch/cv-portfolio
git push origin main
```

- [ ] **Step 5: Esperar build (60-120s) y consultar status vía API**

```bash
set -a; source .env.local; set +a
curl -sS -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/emanuelancco/deployments" \
  | jq '.result[0] | {url, latest_stage: .latest_stage.name, status: .latest_stage.status}'
```

Expected: eventualmente `{"url": "https://....emanuelancco.pages.dev", "latest_stage": "deploy", "status": "success"}`.

- [ ] **Step 6: Smoke test del deploy**

Abrir `https://emanuelancco.pages.dev` en navegador. Expected: ves "Hola." con serif. Footer abajo.

- [ ] **Step 7: Commit (no hay archivos, solo registramos el milestone con tag)**

```bash
git tag -a v0.1-scaffold -m "Astro scaffold deployed to Cloudflare Pages"
git push origin v0.1-scaffold
```

---

## Stage B — Content Collections + migración de proyectos

### Task B1: Definir schema de Content Collection `proyectos`

**Files:**
- Create: `site/src/content.config.ts`

- [ ] **Step 1: Crear `site/src/content.config.ts`**

```ts
import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const proyectos = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/proyectos" }),
  schema: ({ image }) => z.object({
    titulo: z.string(),
    eje: z.enum(["investigacion", "ia-campo", "plataformas", "autoria"]),
    seccion: z.enum(["top12", "archive"]),
    anio: z.number().int().min(2020).max(2030),
    stack: z.array(z.string()).default([]),
    metricas: z.array(z.object({ valor: z.string(), label: z.string() })).default([]),
    enlaces: z.array(z.object({ label: z.string(), url: z.string().url() })).default([]),
    hero: image().optional(),
    galeria: z.array(image()).default([]),
    excerpt: z.string().min(20).max(280),
    orden: z.number().int().default(99),
    publico: z.boolean().default(true),
  }),
});

export const collections = { proyectos };
```

- [ ] **Step 2: Verificar tipos con build**

```bash
cd site && pnpm astro sync && pnpm astro check
```

Expected: 0 errors (la colección está vacía pero el schema compila). Si `astro check` falla por falta de paquete, instalar: `pnpm add -D @astrojs/check typescript`.

- [ ] **Step 3: Commit**

```bash
git add site/src/content.config.ts
git commit -m "feat(content): define proyectos Content Collection schema"
```

---

### Task B2: Migrar 1 proyecto como plantilla — `paper-gnn-puente-junin.md`

**Files:**
- Create: `site/src/content/proyectos/paper-gnn-puente-junin.md`

- [ ] **Step 1: Crear el archivo con frontmatter completo**

```md
---
titulo: "Paper MISM-GNN — Modelo sustituto para el puente Junín"
eje: "investigacion"
seccion: "top12"
anio: 2026
stack: ["PyTorch", "PyTorch Geometric", "OpenSeesPy", "Python", "LaTeX"]
metricas:
  - { valor: "R² 0.998", label: "Precisión GNN" }
  - { valor: "65×", label: "Menos parámetros vs FFNN" }
  - { valor: "15 figs", label: "Generadas para el manuscrito" }
enlaces:
  - { label: "Repositorio (interno)", url: "https://github.com/EmanuelAncco" }
excerpt: "Red neuronal de grafos como modelo sustituto del puente arco Junín. R² 0.998 con 65× menos parámetros que la red feed-forward de referencia. Paper-ready para Elsevier Structures."
orden: 1
---

Modelo sustituto basado en GNN para análisis dinámico estructural del puente arco de Junín, validado contra simulaciones FEM en OpenSees. La GNN respeta la topología del modelo discretizado y aprende propagación de cargas con 4 capas y 65× menos parámetros que la FFNN de referencia, manteniendo R² 0.998 sobre el conjunto de validación.

El paper acompaña 4 tablas y 15 figuras generadas íntegramente desde scripts Python reproducibles.
```

- [ ] **Step 2: Verificar parseo**

```bash
cd site && pnpm astro sync
```

Expected: sin errors. Si falla, leer el mensaje y corregir el frontmatter.

- [ ] **Step 3: Commit**

```bash
git add site/src/content/proyectos/paper-gnn-puente-junin.md
git commit -m "feat(content): migrate paper GNN puente Junín as collection template"
```

---

### Task B3: Migrar los 11 proyectos top12 restantes

**Files:**
- Create: 11 archivos `site/src/content/proyectos/<slug>.md`

Slugs y datos a usar (todos `seccion: "top12"`):

| Slug | titulo | eje | anio | orden |
|---|---|---|---|---|
| `gaiatech-shm` | "GAIATECH SHM — Health monitoring estructural con IA" | investigacion | 2025 | 2 |
| `gaiatech-fpga` | "GAIATECH FPGA — Explorer Edge-9K para FFT en hardware" | investigacion | 2026 | 3 |
| `emarc-vision` | "EMARC VISIÓN — Detección YOLO para seguridad en obra" | ia-campo | 2025 | 4 |
| `genplus-vision-pdk` | "Gen+ Vision PDK — Dashboard YOLOv8 en vivo" | ia-campo | 2026 | 5 |
| `pachaguard` | "PachaGuard — IoT sísmico distribuido" | ia-campo | 2025 | 6 |
| `aecode-finder` | "AECODE FINDER v2 — Panel de empresas con IA + Notion" | plataformas | 2026 | 7 |
| `brochure-studio` | "Brochure Studio — Editor canvas para brochures con IA" | plataformas | 2026 | 8 |
| `email-studio` | "Email Studio — Correos masivos + certificados PDF" | plataformas | 2026 | 9 |
| `aecodito` | "Aecodito v3.0 — Centro de operaciones AECODE" | plataformas | 2026 | 10 |
| `oficina-godot` | "Oficina Virtual AECODE — En Godot 4.4 con MCP" | autoria | 2026 | 11 |
| `emarc-bim-suite` | "EMARC BIM SUITE — Automatización Revit + SAP2000" | autoria | 2025 | 12 |

- [ ] **Step 1: Crear los 11 archivos siguiendo la plantilla de B2**

Cada uno con frontmatter completo + excerpt 20-280 chars + body con 1-2 párrafos extraídos del HTML actual o del Vault Obsidian. **Source de copy:** `legacy/index.html.2026-05-03.bak` (proyectos existentes) o `C:\Users\Emanuel\Obsidian\AECODE\Proyectos\` (proyectos nuevos).

Para cada uno: leer la sección correspondiente en HTML/Obsidian, extraer 1 párrafo descriptivo, llenar `metricas` (al menos 2), `stack` (3-6 items), `enlaces` (1-3), `excerpt` (resumen de 1 línea).

- [ ] **Step 2: Verificar que los 12 parsean**

```bash
cd site && pnpm astro sync
node -e "const fs=require('fs');const f=fs.readdirSync('src/content/proyectos');console.log(f.length,f);"
```

Expected: 12 archivos listados. `astro sync` sin errors.

- [ ] **Step 3: Commit**

```bash
git add site/src/content/proyectos
git commit -m "feat(content): migrate 11 remaining top-12 projects from legacy HTML and Obsidian"
```

---

### Task B4: Crear los 11 proyectos de `/archive` con frontmatter mínimo

**Files:**
- Create: 11 archivos `site/src/content/proyectos/archive/<slug>.md`

Slugs (todos `seccion: "archive"`):

`gaiatech-gestor-ia`, `clawdbot`, `scripts-automatizacion`, `n8n-workflows`, `tuberias-tgd`, `emairc-hp-prime`, `archon-assistant`, `genplus-flows`, `genplus-instructor-finder`, `coord-studio`, `tickets-diplomado-v1`.

- [ ] **Step 1: Crear archivos con frontmatter mínimo (excerpt + 1 métrica + stack)**

Cada uno (ejemplo):

```md
---
titulo: "Suite Tuberías TGD"
eje: "plataformas"
seccion: "archive"
anio: 2024
stack: ["Python", "PyQt", "matplotlib"]
metricas:
  - { valor: "30+ tramos", label: "Cálculo automático" }
excerpt: "Calculadora de pérdidas en tuberías y trazado polar para TGD."
orden: 50
---
```

- [ ] **Step 2: Sync y verificar 23 totales (12 top + 11 archive)**

```bash
cd site && pnpm astro sync
ls site/src/content/proyectos/ -R | grep "\.md$" | wc -l
```

Expected: `23`.

- [ ] **Step 3: Commit**

```bash
git add site/src/content/proyectos/archive
git commit -m "feat(content): add 11 archive projects with minimal frontmatter"
```

---

## Stage C — Componentes base

### Task C1: `ProjectCard.astro` — tarjeta usada en home y archive

**Files:**
- Create: `site/src/components/ProjectCard.astro`

- [ ] **Step 1: Crear el componente**

```astro
---
import type { CollectionEntry } from "astro:content";

interface Props {
  proyecto: CollectionEntry<"proyectos">;
  variant?: "default" | "compact";
}
const { proyecto, variant = "default" } = Astro.props;
const { titulo, anio, stack, metricas, excerpt } = proyecto.data;
const slug = proyecto.id.replace(/\.md$/, "").split("/").pop();
---
<article class={variant === "compact"
  ? "border-b border-[var(--color-rule)] py-6"
  : "border-b border-[var(--color-rule)] py-10"}>
  <div class="flex items-baseline justify-between gap-6">
    <h3 class="font-[var(--font-display)] text-2xl md:text-3xl leading-tight">
      <a href={`/proyectos/${slug}`} class="hover:text-[var(--color-accent)]">{titulo}</a>
    </h3>
    <span class="font-[var(--font-mono)] text-xs text-[var(--color-text-dim)] shrink-0">{anio}</span>
  </div>
  <p class="mt-3 text-[var(--color-text-dim)] max-w-2xl">{excerpt}</p>
  {variant === "default" && metricas.length > 0 && (
    <dl class="mt-4 flex flex-wrap gap-x-8 gap-y-2 font-[var(--font-mono)] text-xs">
      {metricas.map((m) => (
        <div class="flex flex-col">
          <dt class="text-[var(--color-text-dim)] uppercase tracking-wider">{m.label}</dt>
          <dd class="text-[var(--color-accent)]">{m.valor}</dd>
        </div>
      ))}
    </dl>
  )}
  {stack.length > 0 && (
    <div class="mt-3 flex flex-wrap gap-2 text-xs text-[var(--color-text-dim)]">
      {stack.map((s) => <span class="font-[var(--font-mono)]">{s}</span>)}
    </div>
  )}
</article>
```

- [ ] **Step 2: Verificar TypeScript no rompe**

```bash
cd site && pnpm astro check
```

Expected: 0 errors (todavía no se usa, pero compila).

- [ ] **Step 3: Commit**

```bash
git add site/src/components/ProjectCard.astro
git commit -m "feat(components): add ProjectCard with default and compact variants"
```

---

### Task C2: `AxisSection.astro` — sección de un eje en home

**Files:**
- Create: `site/src/components/AxisSection.astro`

- [ ] **Step 1: Crear el componente**

```astro
---
import type { CollectionEntry } from "astro:content";
import ProjectCard from "./ProjectCard.astro";

interface Props {
  label: string;
  titulo: string;
  proyectos: CollectionEntry<"proyectos">[];
}
const { label, titulo, proyectos } = Astro.props;
---
<section class="max-w-[var(--container-1100)] mx-auto px-6 mt-24">
  <header class="border-t border-[var(--color-rule)] pt-6 mb-2">
    <p class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">{label}</p>
    <h2 class="font-[var(--font-display)] text-3xl md:text-4xl mt-2">{titulo}</h2>
  </header>
  {proyectos.map((p) => <ProjectCard proyecto={p} />)}
</section>
```

- [ ] **Step 2: Commit**

```bash
git add site/src/components/AxisSection.astro
git commit -m "feat(components): add AxisSection wrapper for grouped projects"
```

---

### Task C3: `Hero.astro` — hero de home con foto y título

**Files:**
- Create: `site/src/components/Hero.astro`
- Create: `site/src/assets/emanuel.jpg` (copiar desde Descargas)

- [ ] **Step 1: Copiar la foto al repo**

```bash
mkdir -p site/src/assets
cp "/c/Users/Emanuel/Downloads/eab73d94-40d0-4146-b317-8b99d4e58354.jpg" site/src/assets/emanuel.jpg
```

- [ ] **Step 2: Crear `site/src/components/Hero.astro`**

```astro
---
import { Image } from "astro:assets";
import emanuel from "../assets/emanuel.jpg";
---
<section class="max-w-[var(--container-1100)] mx-auto px-6 pt-24 md:pt-32">
  <div class="grid md:grid-cols-[1fr_auto] gap-10 md:gap-16 items-end">
    <div>
      <p class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">Emanuel Ancco · Lima / Puno · 2026</p>
      <h1 class="font-[var(--font-display)] text-5xl md:text-7xl leading-[0.95] mt-4 tracking-tight">
        Construyo en el cruce de la <span class="text-[var(--color-accent)]">ingeniería civil</span>, la <span class="text-[var(--color-accent)]">IA</span> y el <span class="text-[var(--color-accent)]">hardware</span> desde el sur del Perú.
      </h1>
      <p class="mt-6 text-lg text-[var(--color-text-dim)] max-w-xl">
        12 obras validadas en 2026 — investigación, IA aplicada al campo, plataformas operativas y autoría multidisciplinar.
      </p>
    </div>
    <Image
      src={emanuel}
      alt="Retrato de Emanuel Ancco"
      width={320}
      height={400}
      loading="eager"
      class="rounded-md object-cover object-top w-40 md:w-64 border border-[var(--color-rule)]"
    />
  </div>
</section>
```

- [ ] **Step 3: Verificar render**

```bash
cd site && pnpm dev
```

Aún no se monta en index — se verá vacío. Continuamos a C4.

- [ ] **Step 4: Commit**

```bash
git add site/src/assets site/src/components/Hero.astro
git commit -m "feat(components): add Hero with portrait and editorial title"
```

---

### Task C4: `SkillsMarquee.astro` — tira de skills (rescate del actual)

**Files:**
- Create: `site/src/components/SkillsMarquee.astro`

- [ ] **Step 1: Crear el componente**

```astro
---
const skills = [
  "Astro", "TypeScript", "React", "Three.js", "Python",
  "PyTorch", "OpenCV", "YOLOv8", "GNN", "FastAPI",
  "Next.js", "Tailwind", "Postgres", "n8n", "Verilog",
  "FPGA Gowin", "Godot 4", "Blender", "Revit API", "OpenSees",
  "Cloudflare", "Docker", "LaTeX",
];
const doubled = [...skills, ...skills];
---
<section class="overflow-hidden border-y border-[var(--color-rule)] mt-20 py-6">
  <div class="flex gap-10 whitespace-nowrap animate-[marquee_60s_linear_infinite] font-[var(--font-mono)] text-sm text-[var(--color-text-dim)]">
    {doubled.map((s) => <span>{s}</span>)}
  </div>
</section>
<style>
  @keyframes marquee {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add site/src/components/SkillsMarquee.astro
git commit -m "feat(components): add SkillsMarquee with infinite scroll"
```

---

## Stage D — Páginas

### Task D1: Home (`/`) — Hero + 4 ejes + marquee + CTA contacto

**Files:**
- Modify: `site/src/pages/index.astro`

- [ ] **Step 1: Reemplazar `site/src/pages/index.astro`**

```astro
---
import Base from "../layouts/Base.astro";
import Hero from "../components/Hero.astro";
import SkillsMarquee from "../components/SkillsMarquee.astro";
import AxisSection from "../components/AxisSection.astro";
import { getCollection } from "astro:content";

const top12 = (await getCollection("proyectos", (p) => p.data.seccion === "top12" && p.data.publico))
  .sort((a, b) => a.data.orden - b.data.orden);

const ejes = [
  { key: "investigacion", label: "Eje 01", titulo: "Investigación e I+D" },
  { key: "ia-campo",      label: "Eje 02", titulo: "IA aplicada al campo" },
  { key: "plataformas",   label: "Eje 03", titulo: "Plataformas operativas" },
  { key: "autoria",       label: "Eje 04", titulo: "Autoría multidisciplinar" },
] as const;
---
<Base title="Emanuel Ancco — Portfolio">
  <Hero />
  <SkillsMarquee />
  {ejes.map((eje) => (
    <AxisSection
      label={eje.label}
      titulo={eje.titulo}
      proyectos={top12.filter((p) => p.data.eje === eje.key)}
    />
  ))}
  <section class="max-w-[var(--container-1100)] mx-auto px-6 mt-32 border-t border-[var(--color-rule)] pt-16">
    <p class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">Contacto</p>
    <h2 class="font-[var(--font-display)] text-4xl mt-2">Hablemos de un proyecto.</h2>
    <p class="mt-4 text-[var(--color-text-dim)] max-w-xl">
      Email · LinkedIn · GitHub. También leo /archive y /sobre.
    </p>
    <div class="mt-6 flex gap-6 font-[var(--font-mono)] text-sm">
      <a href="/contacto" class="hover:text-[var(--color-accent)]">→ /contacto</a>
      <a href="/archive" class="hover:text-[var(--color-accent)]">→ /archive</a>
      <a href="/cv" class="hover:text-[var(--color-accent)]">→ /cv</a>
      <a href="/sobre" class="hover:text-[var(--color-accent)]">→ /sobre</a>
    </div>
  </section>
</Base>
```

- [ ] **Step 2: Verificar dev**

```bash
cd site && pnpm dev
```

Expected: home con foto + título grande + marquee + 4 ejes con tarjetas. Cada tarjeta tiene título, año, excerpt, métricas, stack. `Ctrl+C`.

- [ ] **Step 3: Commit**

```bash
git add site/src/pages/index.astro
git commit -m "feat(pages): home with hero, marquee, 4 ejes and CTA"
```

---

### Task D2: `/proyectos/[slug]` — página individual de cada proyecto

**Files:**
- Create: `site/src/pages/proyectos/[...slug].astro`

- [ ] **Step 1: Crear la ruta dinámica**

```astro
---
import Base from "../../layouts/Base.astro";
import { getCollection, render } from "astro:content";
import { Image } from "astro:assets";

export async function getStaticPaths() {
  const all = await getCollection("proyectos", (p) => p.data.publico);
  return all.map((proyecto) => ({
    params: { slug: proyecto.id.replace(/\.md$/, "") },
    props: { proyecto },
  }));
}

const { proyecto } = Astro.props;
const { Content } = await render(proyecto);
const { titulo, anio, stack, metricas, enlaces, hero, galeria, excerpt } = proyecto.data;
---
<Base title={`${titulo} — Emanuel Ancco`} description={excerpt}>
  <article class="max-w-[var(--container-1100)] mx-auto px-6 pt-24">
    <p class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">{anio} · {proyecto.data.eje}</p>
    <h1 class="font-[var(--font-display)] text-4xl md:text-6xl mt-3 leading-tight">{titulo}</h1>
    <p class="mt-6 text-xl text-[var(--color-text-dim)] max-w-2xl">{excerpt}</p>
    {hero && (
      <Image src={hero} alt={titulo} class="mt-10 w-full rounded-md border border-[var(--color-rule)]" />
    )}
    {metricas.length > 0 && (
      <dl class="mt-10 grid grid-cols-2 md:grid-cols-4 gap-6 border-y border-[var(--color-rule)] py-8">
        {metricas.map((m) => (
          <div>
            <dt class="font-[var(--font-mono)] text-xs uppercase tracking-wider text-[var(--color-text-dim)]">{m.label}</dt>
            <dd class="font-[var(--font-display)] text-2xl text-[var(--color-accent)] mt-1">{m.valor}</dd>
          </div>
        ))}
      </dl>
    )}
    <div class="prose prose-invert mt-10 max-w-2xl">
      <Content />
    </div>
    {stack.length > 0 && (
      <div class="mt-10">
        <h2 class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">Stack</h2>
        <div class="mt-2 flex flex-wrap gap-3 font-[var(--font-mono)] text-sm">
          {stack.map((s) => <span class="border border-[var(--color-rule)] px-2 py-1 rounded">{s}</span>)}
        </div>
      </div>
    )}
    {enlaces.length > 0 && (
      <div class="mt-10">
        <h2 class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">Enlaces</h2>
        <ul class="mt-2 space-y-1">
          {enlaces.map((e) => (
            <li><a href={e.url} class="text-[var(--color-accent)] hover:underline">{e.label} →</a></li>
          ))}
        </ul>
      </div>
    )}
    {galeria.length > 0 && (
      <div class="mt-12 grid md:grid-cols-2 gap-4">
        {galeria.map((img) => <Image src={img} alt={titulo} class="rounded-md border border-[var(--color-rule)]" />)}
      </div>
    )}
    <p class="mt-16 font-[var(--font-mono)] text-sm">
      <a href="/" class="text-[var(--color-text-dim)] hover:text-[var(--color-accent)]">← Volver al portfolio</a>
    </p>
  </article>
</Base>
```

- [ ] **Step 2: Instalar @tailwindcss/typography para `prose`**

```bash
cd site && pnpm add -D @tailwindcss/typography
```

- [ ] **Step 3: Añadir plugin en `globals.css` (después del `@import`)**

```css
@plugin "@tailwindcss/typography";
```

- [ ] **Step 4: Verificar build**

```bash
cd site && pnpm build
```

Expected: build exitoso, `dist/proyectos/<slug>/index.html` generado para cada proyecto. Si falla por slug con `/` (de archive/), revisar `getStaticPaths` — el `[...slug]` rest soporta subdirs.

- [ ] **Step 5: Commit**

```bash
git add site/
git commit -m "feat(pages): dynamic project pages at /proyectos/[slug]"
```

---

### Task D3: `/archive` — grilla compacta de los 11 secundarios

**Files:**
- Create: `site/src/pages/archive.astro`

- [ ] **Step 1: Crear**

```astro
---
import Base from "../layouts/Base.astro";
import ProjectCard from "../components/ProjectCard.astro";
import { getCollection } from "astro:content";

const archive = (await getCollection("proyectos", (p) => p.data.seccion === "archive" && p.data.publico))
  .sort((a, b) => b.data.anio - a.data.anio || a.data.orden - b.data.orden);
---
<Base title="Archivo — Emanuel Ancco" description="Proyectos secundarios validados en el archivo de Emanuel Ancco.">
  <section class="max-w-[var(--container-1100)] mx-auto px-6 pt-24">
    <p class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">Archivo</p>
    <h1 class="font-[var(--font-display)] text-4xl md:text-6xl mt-3">Trabajo secundario, también real.</h1>
    <p class="mt-6 text-[var(--color-text-dim)] max-w-2xl">
      Proyectos que existen, funcionan o se publicaron, pero no representan la apuesta principal de cada eje. Listados por año.
    </p>
    <div class="mt-12">
      {archive.map((p) => <ProjectCard proyecto={p} variant="compact" />)}
    </div>
  </section>
</Base>
```

- [ ] **Step 2: Verificar dev en `/archive`**

```bash
cd site && pnpm dev
```

Expected: 11 tarjetas compactas listadas. `Ctrl+C`.

- [ ] **Step 3: Commit**

```bash
git add site/src/pages/archive.astro
git commit -m "feat(pages): /archive with 11 secondary validated projects"
```

---

### Task D4: `/cv` — página con 1 CV (el más actualizado)

**Files:**
- Create: `site/src/pages/cv.astro`
- Create: `site/public/cv/CV_GenPlus_IA_2026.pdf`

Decisión del usuario: solo se publica 1 CV — el más reciente (`CV_GenPlus_IA_2026.tex`, 17-mar-2026). Los otros 3 .tex permanecen en repo pero no se publican.

- [ ] **Step 1: Compilar 1 .tex a PDF**

Si `tectonic` está instalado (recomendado, single-binary):

```bash
mkdir -p site/public/cv
tectonic -o site/public/cv CV_GenPlus_IA_2026.tex
ls -lh site/public/cv/
```

Si `tectonic` no está, usar `latexmk` (TeX Live):

```bash
mkdir -p site/public/cv
latexmk -pdf -output-directory=site/public/cv CV_GenPlus_IA_2026.tex
ls -lh site/public/cv/
```

Expected: 1 PDF entre 100-500 KB.

- [ ] **Step 2: Crear `site/src/pages/cv.astro`**

```astro
---
import Base from "../layouts/Base.astro";

const cv = {
  archivo: "CV_GenPlus_IA_2026.pdf",
  titulo: "Currículum — versión actualizada",
  fecha: "Marzo 2026",
  para: "Resume mi trayectoria en ingeniería civil, IA aplicada al campo, hardware y plataformas operativas. Versión vigente.",
};
---
<Base title="CV — Emanuel Ancco" description="Currículum PDF actualizado de Emanuel Ancco.">
  <section class="max-w-[var(--container-1100)] mx-auto px-6 pt-24">
    <p class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">Currículum · {cv.fecha}</p>
    <h1 class="font-[var(--font-display)] text-4xl md:text-6xl mt-3 leading-tight">Una sola página resume el resto.</h1>
    <p class="mt-6 text-[var(--color-text-dim)] max-w-2xl">{cv.para}</p>
    <div class="mt-12">
      <a href={`/cv/${cv.archivo}`} class="inline-block border border-[var(--color-rule)] px-8 py-6 hover:border-[var(--color-accent)] transition-colors">
        <h2 class="font-[var(--font-display)] text-2xl">{cv.titulo}</h2>
        <p class="mt-3 font-[var(--font-mono)] text-sm text-[var(--color-accent)]">Descargar PDF →</p>
      </a>
    </div>
    <p class="mt-12 text-sm text-[var(--color-text-dim)] max-w-2xl">
      Para versiones específicas (Las Bambas Geotecnia, AECODE Producto, etc.), escríbeme directo desde
      <a href="/contacto" class="text-[var(--color-accent)] hover:underline">/contacto</a>.
    </p>
  </section>
</Base>
```

- [ ] **Step 3: Verificar dev**

Expected: ver 4 cards en `/cv`, click descarga el PDF correcto.

- [ ] **Step 4: Commit**

```bash
git add site/public/cv site/src/pages/cv.astro
git commit -m "feat(pages): /cv with 4 PDF versions for distinct audiences"
```

---

### Task D5: `/sobre` — bio editorial

**Files:**
- Create: `site/src/pages/sobre.astro`

- [ ] **Step 1: Crear**

```astro
---
import Base from "../layouts/Base.astro";
---
<Base title="Sobre — Emanuel Ancco" description="Quién es Emanuel Ancco — ingeniero civil, constructor de IA y hardware desde el sur del Perú.">
  <article class="max-w-[var(--container-1100)] mx-auto px-6 pt-24">
    <p class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">Sobre</p>
    <h1 class="font-[var(--font-display)] text-4xl md:text-6xl mt-3 leading-tight">
      Soy ingeniero civil, pero mi día se parece más al de un constructor de cosas que aún no existen.
    </h1>

    <div class="prose prose-invert mt-12 max-w-2xl">
      <p>
        Vengo del sur del Perú — Puno, después Lima. Estudié ingeniería civil porque me interesaban
        los puentes, las estructuras y la matemática de cómo el mundo aguanta su peso. Terminé
        construyendo otra cosa: un cruce entre ingeniería, IA aplicada al campo, hardware embebido
        y plataformas internas para los equipos en los que trabajo.
      </p>
      <h2>Cómo trabajo</h2>
      <p>
        Soy multidisciplinar por necesidad. En el sur del Perú no hay equipos completos, hay
        problemas completos. Aprendí PyTorch porque nadie iba a entrenar el modelo de visión que
        necesitaba el proyecto de Gen+. Aprendí Verilog porque la FFT en software no daba el
        ancho de banda que necesitaba GAIATECH. Aprendí Astro porque el portfolio anterior ya no
        contaba quién soy.
      </p>
      <h2>Qué busco colaborar</h2>
      <p>
        Me interesan los problemas raros: monitoreo estructural con IA, edge AI en obra, FPGAs
        para señal, plataformas operativas que reemplazan procesos manuales. Y los proyectos que
        miran al sur — al campo, a las comunidades, a los problemas que no salen en TechCrunch.
      </p>
      <h2>Lo que no soy</h2>
      <p>
        No soy especialista profundo de un solo stack. No vendo lo que no he construido. No
        compito con la gente que admiro — colaboro. Si lo que ves en este portfolio te interesa,
        escríbeme.
      </p>
    </div>

    <p class="mt-16 font-[var(--font-mono)] text-sm">
      <a href="/contacto" class="text-[var(--color-accent)] hover:underline">→ Contacto</a>
    </p>
  </article>
</Base>
```

- [ ] **Step 2: Commit**

```bash
git add site/src/pages/sobre.astro
git commit -m "feat(pages): /sobre with editorial bio"
```

---

### Task D6: `/contacto` — links + email ofuscado

**Files:**
- Create: `site/src/pages/contacto.astro`

- [ ] **Step 1: Crear**

```astro
---
import Base from "../layouts/Base.astro";
const emailUser = "coarp.eancco";
const emailDomain = "gmail.com";
---
<Base title="Contacto — Emanuel Ancco">
  <section class="max-w-[var(--container-1100)] mx-auto px-6 pt-24">
    <p class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">Contacto</p>
    <h1 class="font-[var(--font-display)] text-4xl md:text-6xl mt-3">Hablemos.</h1>
    <p class="mt-6 text-[var(--color-text-dim)] max-w-2xl">
      Email es la vía más rápida. LinkedIn y GitHub también funcionan.
    </p>
    <ul class="mt-10 space-y-4 font-[var(--font-mono)] text-lg">
      <li>
        Email · <a class="text-[var(--color-accent)] hover:underline" href={`mailto:${emailUser}@${emailDomain}`}>{emailUser}<span aria-hidden="true">[at]</span>{emailDomain}</a>
      </li>
      <li>
        LinkedIn · <a class="text-[var(--color-accent)] hover:underline" href="https://www.linkedin.com/in/emanuel-ancco/">linkedin.com/in/emanuel-ancco</a>
      </li>
      <li>
        GitHub · <a class="text-[var(--color-accent)] hover:underline" href="https://github.com/EmanuelAncco">github.com/EmanuelAncco</a>
      </li>
    </ul>
  </section>
</Base>
```

- [ ] **Step 2: Commit**

```bash
git add site/src/pages/contacto.astro
git commit -m "feat(pages): /contacto with obfuscated email and links"
```

---

### Task D7: 404 + sitemap + robots

**Files:**
- Create: `site/src/pages/404.astro`
- Modify: `site/astro.config.mjs`
- Create: `site/public/robots.txt`

- [ ] **Step 1: Instalar `@astrojs/sitemap`**

```bash
cd site && pnpm add @astrojs/sitemap
```

- [ ] **Step 2: Actualizar `site/astro.config.mjs`**

```js
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://emanuelancco.com',
  integrations: [sitemap()],
  vite: { plugins: [tailwindcss()] },
});
```

- [ ] **Step 3: Crear `site/public/robots.txt`**

```
User-agent: *
Allow: /

Sitemap: https://emanuelancco.com/sitemap-index.xml
```

- [ ] **Step 4: Crear `site/src/pages/404.astro`**

```astro
---
import Base from "../layouts/Base.astro";
---
<Base title="404 — Emanuel Ancco">
  <section class="max-w-[var(--container-1100)] mx-auto px-6 pt-32">
    <p class="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">Error 404</p>
    <h1 class="font-[var(--font-display)] text-5xl md:text-7xl mt-3">No encontrado.</h1>
    <p class="mt-6 text-[var(--color-text-dim)]">Esa ruta no existe (todavía). Volver al <a href="/" class="text-[var(--color-accent)] hover:underline">portfolio</a>.</p>
  </section>
</Base>
```

- [ ] **Step 5: Build y verificar**

```bash
cd site && pnpm build
ls dist/sitemap*.xml dist/robots.txt dist/404.html
```

Expected: existen.

- [ ] **Step 6: Commit**

```bash
git add site/
git commit -m "feat(seo): add sitemap, robots.txt and 404 page"
```

---

## Stage E — Migración de assets e imágenes

### Task E1: Mover y optimizar imágenes existentes

**Files:**
- Move: `assets/images/*` → `site/src/assets/projects/`
- Update: archivos `.md` afectados con frontmatter `hero` y `galeria`

- [ ] **Step 1: Mover imágenes a `site/src/assets/projects/`**

```bash
mkdir -p site/src/assets/projects
cp -r assets/images/* site/src/assets/projects/
ls site/src/assets/projects/ | head -20
```

- [ ] **Step 2: Vincular hero/galería en los proyectos correspondientes**

Editar los 12 .md de top12 (cuando aplique) añadiendo en frontmatter:

```yaml
hero: ../../assets/projects/<archivo>.png
galeria:
  - ../../assets/projects/<otro>.png
```

Ejemplo concreto para `gaiatech-shm.md`:

```yaml
hero: ../../assets/projects/gaiatech_results.png
galeria:
  - ../../assets/projects/gaiatech_model_comparison.png
  - ../../assets/projects/neural_network_architecture.png
```

Y para `emarc-vision.md`:

```yaml
hero: ../../assets/projects/emarc_vision_detection.png
galeria:
  - ../../assets/projects/BoxF1_curve.png
  - ../../assets/projects/confusion_matrix.png
```

Y para `emarc-bim-suite.md`:

```yaml
hero: ../../assets/projects/revit_automation.png
```

Para los proyectos sin imagen propia (los nuevos como `aecode-finder`, `brochure-studio`, `aecodito`), generar capturas reales con DevTools del panel funcional o dejar sin `hero` (el componente lo renderiza condicional).

- [ ] **Step 3: Build y verificar que no hay broken images**

```bash
cd site && pnpm build 2>&1 | grep -i "error\|warn" | head -20
```

Expected: 0 errores. Warnings de imagen aceptables (formato sin alpha, etc).

- [ ] **Step 4: Commit**

```bash
git add site/src/assets/projects site/src/content/proyectos
git commit -m "feat(assets): migrate project images and link as hero/gallery in frontmatter"
```

---

### Task E2: OG image por defecto + favicon

**Files:**
- Create: `site/public/og-default.jpg`
- Create: `site/public/favicon.svg`

- [ ] **Step 1: Generar OG image (1200×630) — versión rápida en HTML→imagen**

Crear `tmp-og.html` con tu foto + nombre + url, screenshot a 1200×630 con DevTools o Playwright (un solo comando):

```bash
cat > /tmp/og.html <<'EOF'
<!doctype html><html><head><style>
body{margin:0;width:1200px;height:630px;background:#020617;color:#f8fafc;display:flex;align-items:center;justify-content:space-between;padding:80px;box-sizing:border-box;font-family:Georgia,serif}
.t{max-width:700px}
.t h1{font-size:72px;line-height:1;margin:0 0 24px}
.t p{font-size:24px;color:#94a3b8;margin:0}
img{width:300px;height:380px;object-fit:cover;object-position:top;border-radius:8px}
</style></head><body>
<div class="t"><h1>Emanuel Ancco</h1><p>Ingeniería civil · IA · Hardware<br/>desde el sur del Perú</p><p style="color:#f59e0b;margin-top:24px">emanuelancco.com</p></div>
<img src="file:///c:/Users/Emanuel/Downloads/eab73d94-40d0-4146-b317-8b99d4e58354.jpg"/>
</body></html>
EOF
```

Capturar a 1200x630 con Chromium headless (si no, screenshot manual sirve):

```bash
chromium --headless --screenshot=site/public/og-default.jpg --window-size=1200,630 file:///tmp/og.html
```

Si `chromium` no está, usar Playwright `npx playwright screenshot --viewport-size=1200,630 file:///tmp/og.html site/public/og-default.jpg`.

Si nada funciona: el usuario captura manual con DevTools (zoom 100%, viewport 1200×630, Cmd/Ctrl+Shift+P → "Capture full size screenshot") y guarda como `site/public/og-default.jpg`.

- [ ] **Step 2: Verificar tamaño 1200x630**

```bash
file site/public/og-default.jpg
```

Expected: muestra `1200x630`.

- [ ] **Step 3: Crear favicon SVG simple (letra E con accent dorado)**

```bash
cat > site/public/favicon.svg <<'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" fill="#020617"/>
  <text x="50%" y="50%" text-anchor="middle" dominant-baseline="central" font-family="Georgia, serif" font-size="40" font-weight="700" fill="#f59e0b">E</text>
</svg>
EOF
```

- [ ] **Step 4: Añadir favicon en `Base.astro` head**

Editar `site/src/layouts/Base.astro` y añadir antes del `<title>`:

```astro
<link rel="icon" type="image/svg+xml" href="/favicon.svg" />
```

- [ ] **Step 5: Commit**

```bash
git add site/public site/src/layouts/Base.astro
git commit -m "feat(seo): add default OG image and favicon"
```

---

## Stage F — Polish, dominio y deploy final

### Task F1: Structured data (Person schema) en home

**Files:**
- Modify: `site/src/pages/index.astro`

- [ ] **Step 1: Añadir bloque JSON-LD al final del Base slot en `index.astro`**

Insertar antes del `</Base>` final:

```astro
<script type="application/ld+json" set:html={JSON.stringify({
  "@context": "https://schema.org",
  "@type": "Person",
  "name": "Emanuel Ancco",
  "url": "https://emanuelancco.com",
  "jobTitle": "Ingeniero Civil · Desarrollador IA · Hardware",
  "address": { "@type": "PostalAddress", "addressLocality": "Lima", "addressRegion": "Lima", "addressCountry": "PE" },
  "sameAs": ["https://github.com/EmanuelAncco", "https://www.linkedin.com/in/emanuel-ancco/"]
})}></script>
```

- [ ] **Step 2: Validar con build**

```bash
cd site && pnpm build && grep -c "Emanuel Ancco" dist/index.html
```

Expected: número > 0.

- [ ] **Step 3: Commit**

```bash
git add site/src/pages/index.astro
git commit -m "feat(seo): add Person JSON-LD on home"
```

---

### Task F2: Audit Lighthouse y arreglar issues hasta cumplir métricas

**Files:** los que el audit señale.

Métricas objetivo del spec:
- Performance ≥ 95 (mobile)
- Accessibility ≥ 100
- Best Practices ≥ 100
- SEO ≥ 100
- LCP < 2.0s

- [ ] **Step 1: Servir build local y correr Lighthouse CLI**

```bash
cd site && pnpm build && pnpm preview &
sleep 3
npx lighthouse http://localhost:4321 --only-categories=performance,accessibility,best-practices,seo --form-factor=mobile --throttling.cpuSlowdownMultiplier=4 --output=html --output-path=/tmp/lh.html --chrome-flags="--headless"
kill %1 2>/dev/null
```

Expected: `/tmp/lh.html` generado. Abrirlo y revisar scores.

- [ ] **Step 2: Anotar issues e iterar**

Issues comunes y fixes:
- Faltan `alt` en imágenes → añadir
- Contraste insuficiente → ajustar colores
- LCP alto por hero image → asegurar `loading="eager"` y `fetchpriority="high"` en el `<Image>` del hero (modificar Hero.astro)
- Bundle JS alto → usar `client:visible` o quitar islas innecesarias (en v1 sin /lab no debería haber)

Aplicar fixes hasta llegar a metas. Cada fix = 1 commit pequeño con mensaje `perf:`, `a11y:` o `fix:` según corresponda.

- [ ] **Step 3: Commit final del audit**

```bash
git add site/
git commit -m "perf: lighthouse audit fixes (LCP, alt text, contrast)"
```

---

### Task F3: Conectar dominio `emanuelancco.com` al proyecto Pages vía API

**Files:** ninguno (operación remota).

- [ ] **Step 1: Obtener Zone ID de `emanuelancco.com`**

```bash
set -a; source .env.local; set +a
ZONE_ID=$(curl -sS -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/zones?name=emanuelancco.com" | jq -r '.result[0].id')
echo "ZONE_ID=$ZONE_ID"
```

Expected: 32-char hex. Guardar en `.env.local` como `CLOUDFLARE_ZONE_ID=...`.

- [ ] **Step 2: Adjuntar dominio apex al proyecto Pages**

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/emanuelancco/domains" \
  -d '{"name": "emanuelancco.com"}' | jq .
```

Expected: `{"result":{"name":"emanuelancco.com","status":"pending",...},"success":true}`.

- [ ] **Step 3: Adjuntar `www.` también**

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/emanuelancco/domains" \
  -d '{"name": "www.emanuelancco.com"}' | jq .
```

- [ ] **Step 4: Verificar que Cloudflare creó automáticamente los DNS records (CNAME a `emanuelancco.pages.dev`)**

```bash
curl -sS -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?name=emanuelancco.com" | jq '.result[]|{name,type,content,proxied}'
```

Expected: aparece un CNAME (o flat A) apuntando a `emanuelancco.pages.dev`. Si no aparece, crearlo manualmente:

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" \
  -d '{"type":"CNAME","name":"@","content":"emanuelancco.pages.dev","proxied":true}'

curl -sS -X POST \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" \
  -d '{"type":"CNAME","name":"www","content":"emanuelancco.pages.dev","proxied":true}'
```

- [ ] **Step 5: Esperar SSL (5-15 min) y smoke test**

```bash
curl -I https://emanuelancco.com
```

Expected: `HTTP/2 200`, `server: cloudflare`. Si da 525/526 (SSL handshake), esperar 5 min más.

- [ ] **Step 6: Verificar redirect www → apex**

Por defecto Cloudflare Pages sirve ambos. Para forzar `www.emanuelancco.com` → `emanuelancco.com`, crear Page Rule o Redirect Rule:

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/rulesets" \
  -d '{
    "name":"redirect-www-to-apex",
    "kind":"zone",
    "phase":"http_request_dynamic_redirect",
    "rules":[{
      "expression":"http.host eq \"www.emanuelancco.com\"",
      "action":"redirect",
      "action_parameters":{
        "from_value":{
          "status_code":301,
          "target_url":{"expression":"concat(\"https://emanuelancco.com\", http.request.uri.path)"},
          "preserve_query_string":true
        }
      }
    }]
  }' | jq .
```

- [ ] **Step 7: Commit milestone con tag**

```bash
git tag -a v1.0 -m "Portfolio v1 live at emanuelancco.com"
git push origin v1.0
```

---

### Task F4: Smoke test final en producción

**Files:** ninguno.

- [ ] **Step 1: Checklist manual sobre `https://emanuelancco.com`**

- [ ] Home carga, hero muestra foto y título, fuentes Fraunces se ven
- [ ] Marquee gira
- [ ] 4 ejes muestran sus 3 proyectos cada uno = 12
- [ ] Click en un proyecto → página `/proyectos/<slug>` carga
- [ ] `/archive` muestra 11 tarjetas
- [ ] `/cv` lista 4 PDFs descargables, todos abren
- [ ] `/sobre` carga
- [ ] `/contacto` muestra email LinkedIn GitHub
- [ ] `/no-existe` muestra 404
- [ ] OG card preview en https://www.opengraph.xyz/url/https%3A%2F%2Femanuelancco.com — se ve la imagen
- [ ] DevTools mobile + desktop: layout sin overflow horizontal
- [ ] Lighthouse mobile ≥ 95/100/100/100

- [ ] **Step 2: Si todo OK, anunciar en LinkedIn / WhatsApp Gen+ (acción humana)**

- [ ] **Step 3: Cerrar plan**

Plan A completado. Listo para empezar Plan B (/lab) y Plan C (copy editorial pass) cuando tú decidas.

---

## Self-review (chequeado contra el spec)

**Spec coverage:**
- Stack Astro+Tailwind ✓ (A2, A3)
- Cinematic dark + tipografía editorial ✓ (A4 fonts, A5 layout, C3 hero)
- 6 rutas (/, /lab, /archive, /cv, /sobre, /contacto) ✓ — /lab queda en plan B (declarado)
- Top 12 en 4 ejes ✓ (B2-B3, D1)
- /archive 11 ✓ (B4, D3)
- /cv 4 PDFs ✓ (D4)
- Migración assets ✓ (E1)
- OG + favicon + sitemap + robots + 404 ✓ (D7, E2)
- Cloudflare Pages + custom domain ✓ (A7, F3)
- SSL + redirects ✓ (F3)
- Lighthouse target ✓ (F2)
- Sin /lab, sin GSAP, sin i18n, sin formulario ✓ (declarado en spec)

**Placeholder scan:** sin TODO/TBD/FIXME en el plan. Donde el spec dice "opcional" lo declaré en la decisión correspondiente.

**Type consistency:** schema Zod en `content.config.ts` usa los mismos campos que ProjectCard, AxisSection, getCollection y la página `[slug]`. `eje` es enum cerrado, los 4 valores se reusan.

**Brechas detectadas e integradas:** ninguna que requiera tarea adicional. Diagramas Mermaid quedan fuera de v1 por decisión del spec.
