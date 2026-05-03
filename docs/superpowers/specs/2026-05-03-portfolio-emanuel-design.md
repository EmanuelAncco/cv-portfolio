# Design — Portfolio Emanuel Ancco v2

**Fecha:** 2026-05-03
**Repo:** [EmanuelAncco/cv-portfolio](https://github.com/EmanuelAncco/cv-portfolio)
**Dominio:** emanuelancco.com (comprado en Cloudflare Registrar, expira 2027-05-03)
**Estado:** spec aprobado, pendiente plan de implementación

## Objetivo

Reemplazar el portfolio actual (single-file `index.html` de 68KB, dark + golden glow + diagramas Mermaid) por un sitio Astro estático que posicione a Emanuel como **marca personal multidisciplinaria** en el cruce Ingeniería Civil + IA + IoT + Hardware + Emprendimiento.

Audiencia primaria: pares técnicos y diseñadoras del entorno Gen+/AECODE que entienden el lenguaje de portafolio editorial. Audiencia secundaria: reclutadores/clientes que llegan por LinkedIn o paper publicado.

No es CV estricto (existen los 4 PDFs LaTeX para eso). Es la página que debe transmitir identidad y nivel técnico al tercer segundo de scroll.

## Decisiones cerradas

| Eje | Valor |
|---|---|
| Estética dominante | Cinematic dark (foto grande, tipografía editorial light, contraste alto, acentos dorados ya existentes) |
| Tono de copy | Editorial / anuario (narra el año, no lista tareas) |
| Wow técnico | 1 detalle 3D contenido en `/lab` (no en hero — coste/beneficio no compensa) |
| Stack | Astro 5 + Tailwind 4 + islas React solo donde haga falta interactividad |
| 3D engine en islas | React Three Fiber + drei (cuando aplique en /lab) |
| Idioma v1 | Español únicamente. Estructura preparada para `i18n` Astro nativo en v2. |
| Hosting | Cloudflare Pages, build automático desde rama `main` del repo |
| URL provisional | `emanuelancco.pages.dev` |
| URL final | `emanuelancco.com` + `www.emanuelancco.com` (custom domain conectado vía Cloudflare Pages) |

## Criterio de selección de proyectos

Se incluyen únicamente proyectos con **documentación validada del trabajo realizado**: entregable verificable + capturas/métricas/datos reales. Se excluyen ideas, planes, postulaciones no adjudicadas, propuestas en preparación y trabajo confidencial bajo NDA.

### Excluidos (para referencia y para evitar reincorporarlos por error)

AECODE Live, Ticket Studio v2, Aecodito robot físico, GAIATECH G1 PROCIENCIA, GAIATECH M1 Las Bambas, Gen+ cámaras Edge AI, Ainy Inmobiliaria (NDA), AECODE Operaciones, Publish Studio (estado ambiguo), Tesis Sheyla.

## Arquitectura de información

### Rutas

```
/              Hero + Top 12 + tira de skills + CTA contacto
/lab           3 demos vivas interactivas
/archive       11 proyectos secundarios validados (tarjetas compactas)
/cv            Los 4 PDFs LaTeX descargables + preview
/sobre         Bio editorial larga (filosofía SOUL.md condensada)
/contacto      Email + LinkedIn + GitHub + WhatsApp Gen+/AECODE (ofuscado)
```

### Top 12 (homepage `/`, agrupados por eje)

| Eje | Proyecto | Evidencia que lo valida |
|---|---|---|
| Investigación / I+D | Paper MISM-GNN Puente Junín | Paper-ready 4 tablas + 15 figs PDF, R² 0.998 GNN vs 0.999 FFNN con 65× menos parámetros, listo para Elsevier Structures |
| Investigación / I+D | GAIATECH SHM | Métricas + imágenes ya en portfolio actual |
| Investigación / I+D | GAIATECH FPGA Explorer Edge-9K | 5 módulos para FFT, Demo Day 18-abr realizada, investigación 253 KB |
| IA aplicada al campo | EMARC VISIÓN | Box curves YOLO + confusion matrices + EMARC vision detection imgs |
| IA aplicada al campo | Gen+ Vision PDK | Live en VPS :3020 dashboard + :8088 FastAPI YOLOv8, informe HTML institucional 7 págs con capturas reales |
| IA aplicada al campo | PachaGuard IoT & Seismic | Existente con doc en portfolio actual |
| Plataformas operativas | AECODE FINDER v2 | Panel Next.js :3002 con sync en vivo Notion BD-Empresas (99 filas) |
| Plataformas operativas | Brochure Studio | Canvas editor Fabric.js :3500 funcional |
| Plataformas operativas | Email Studio + Certificados | Módulo /certificados con pdf-lib estampa nombre + envío masivo Gmail |
| Plataformas operativas | Aecodito v3.0 Centro de Operaciones | 50 nodos n8n, 10 tools, 23 funcionalidades, en uso diario |
| Autoría / multidisciplina | Oficina Virtual AECODE en Godot | Godot 4.4.1, MCP configurado, AECODITO jugable |
| Autoría / multidisciplina | EMARC BIM SUITE | Revit automation existente |

### `/archive` (tarjetas compactas, validados pero secundarios)

GAIATECH Gestor IA · Clawdbot & Agentes Sociales · Scripts de Automatización · Workflows n8n · Suite Tuberías TGD · EMAIRC HIDRA HP Prime · Archon Assistant · Gen+ FLOWS Activepieces · Gen+ Instructor Finder · Coord Studio · Tickets Diplomado AECODE v1.

### `/lab` — 3 demos vivas

| Demo | Tecnología | Datos / fuente |
|---|---|---|
| Visualizador GNN Puente Junín | R3F (escena 3D del puente) + carga animada + tabla R² | Paper MISM-GNN ya generado: tablas + 15 figuras |
| Aecodito chatbot embebido | iframe o widget al endpoint del workflow n8n existente | Aecodito v3.0 ya operacional (50 nodos) |
| Live YOLO Gen+ Vision PDK | iframe / proxy al :8088 del VPS o snapshots periódicos | Stream YOLOv8 ya corriendo en VPS AECODE |

Si el live YOLO comprime mal o tiene latencia inaceptable, fallback a galería de capturas reales con timestamp + descripción del frame.

### `/cv`

**1 solo CV publicado:** `CV_GenPlus_IA_2026.tex` (el más reciente, 17-mar-2026, alineado con el contexto del portfolio). Se compila a PDF y se sirve estático con preview thumbnail + botón descargar.

Los otros 3 .tex (`CV_AECODE_2026`, `CV_Las_Bambas_2026`, `CV_Las_Bambas_Geotecnia_2026`) permanecen en el repo como histórico/uso privado, pero **no se publican** en el sitio.

## Estructura de carpetas (Astro)

```
src/
  layouts/
    Base.astro            Layout raíz (head, fonts, footer, theme)
    Editorial.astro       Variant para /sobre y posts si llegan
  components/
    Hero.astro
    ProjectCard.astro     Tarjeta usada en home y /archive
    AxisSection.astro     Sección de un eje en home
    SkillsMarquee.astro   Tira de skills (rescate del marquee actual)
    Timeline.astro        Experiencia (rescate del timeline actual)
    Footer.astro
    lab/
      GnnBridge.tsx       Isla React + R3F
      AecoditoChat.tsx    Isla React (iframe/widget)
      VisionLive.tsx      Isla React (iframe + fallback)
  pages/
    index.astro
    lab.astro
    archive.astro
    cv.astro
    sobre.astro
    contacto.astro
  content/
    proyectos/            Markdown por proyecto, frontmatter tipado con Astro Content Collections
      paper-gnn-puente-junin.md
      gaiatech-shm.md
      gaiatech-fpga.md
      emarc-vision.md
      genplus-vision-pdk.md
      pachaguard.md
      aecode-finder.md
      brochure-studio.md
      email-studio.md
      aecodito.md
      oficina-godot.md
      emarc-bim-suite.md
      ... (11 más en /archive)
  styles/
    globals.css           Tailwind base + custom props (golden accent, dark bg)
public/
  fonts/                  Self-hosted (no Google Fonts CDN para LCP rápido)
  cv/                     PDFs compilados
  img/                    Imágenes optimizadas (rescate de assets/images existentes)
astro.config.mjs
tailwind.config.mjs
package.json
```

### Por qué Content Collections

Cada proyecto vive en un Markdown con frontmatter tipado (título, eje, año, stack, métricas, links, hero image, gallery). Eso permite:

- Renderizar tarjetas y páginas individuales sin duplicar copy
- Tipado estricto (Astro valida el frontmatter al build)
- Añadir/editar un proyecto = editar un `.md`, no tocar componentes
- Filtrar por eje en home y por categoría en /archive sin código nuevo

## Sistema visual

### Tipografía

- **Display (títulos h1, h2 hero):** Fraunces Variable (serif moderna con peso variable, gratis, OFL) o GT Sectra si hay licencia. Self-hosted.
- **Body:** Inter Variable. Self-hosted.
- **Mono (métricas, código, labels técnicos):** JetBrains Mono Variable.

Justificación: el actual usa Outfit (sans geométrica). Para el feel editorial necesitamos serif en display. Inter para body es la apuesta segura sin perder modernidad.

### Color

Se rescata la paleta dark + golden ya existente:
- `--bg`: #020617 (slate 950, ya en uso)
- `--surface`: #0f172a (slate 900)
- `--text`: #f8fafc
- `--text-dim`: #94a3b8
- `--accent`: #f59e0b → #fbbf24 (gradient golden, ya en uso)
- `--rule`: rgba(255,255,255,0.08) (líneas sutiles, estilo magazine)

Nuevo respecto al actual: introducción de **filetes y reglas tipográficas** (líneas finas tipo NYT) entre secciones, en vez de cards con border-radius grande. Esto baja el "tono dashboard" y sube el "tono publicación".

### Espaciado

Grilla editorial 12 cols. Ancho máximo de contenido: 1100px (más estrecho que el actual 1400px) — fuerza líneas más cortas, lectura más cómoda, sensación de "página" en vez de "dashboard".

## Página por página

### `/` Home

1. **Hero**
   - Foto retrato grande (la que el usuario subió a Descargas, optimizada y servida desde `/img/emanuel.jpg` con `<Image>` de Astro)
   - Título display: una frase corta, no genérica. Ejemplo de partida (a refinar en copy review): *"Construyo en el cruce de la ingeniería civil, la IA y el hardware desde el sur del Perú."*
   - Subline editorial: lo que está haciendo este año (1-2 líneas)
   - Sin tagline tipo "I'm a developer"
2. **Skills marquee** (rescate del actual, sutil)
3. **4 ejes con sus 3 proyectos cada uno**, cada eje como sección independiente con regla tipográfica + label de eje + 3 cards
4. **Métricas del año** (mini-banner con cifras: nº de proyectos validados, nº de líneas de código, nº de modelos entrenados, etc — opcional, decisión de copy en revisión)
5. **CTA contacto** (no formulario; email directo + LinkedIn + GitHub)

### `/lab`

3 demos como secciones verticales scrollables, una por demo. Cada una: título + 1 párrafo de contexto + componente isla + link al proyecto fuente.

### `/archive`

Grilla compacta de 11 tarjetas. Filtro opcional por eje (decisión: en v1 sin filtro, KISS).

### `/cv`

4 cards, cada una con thumbnail PDF + título de la versión + para qué audiencia + botón "Descargar PDF". Si Astro PDF preview es estable, embed inline. Si no, descarga directa.

### `/sobre`

Bio editorial larga, primera persona, tono SOUL.md condensado. 600-900 palabras. Subsecciones con reglas: origen (Puno/sur), trayectoria, filosofía, qué busca colaborar.

### `/contacto`

Email ofuscado + LinkedIn + GitHub + (opcional) calendario. Sin formulario en v1 (no hay backend, no añadimos Resend/etc).

## Migración del contenido existente

El `index.html` actual contiene copy útil de 11 proyectos y ~30 imágenes. Plan de rescate:

1. **Imágenes:** mover `assets/images/` → `public/img/`, optimizar con `astro:assets` (responsive, webp, lazy)
2. **Copy:** extraer descripciones de proyecto del HTML actual y verterlas a los `.md` correspondientes en `content/proyectos/`. Reescritura editorial al portar (no copy-paste literal — hay que adaptar al nuevo tono)
3. **Mermaid diagrams:** los 3 diagramas (LLM map, SAP2000→Revit, Unity) → decisión: o se rescatan en `/lab` como sección de "ecosistema técnico", o se descartan. Recomendación: rescatar como página `/ecosistema` opcional v2; no incluir en v1 para no dispersar el foco.
4. **Timeline experiencia:** rescatar como componente en `/sobre`
5. **CV LaTeX:** dejar los `.tex` en el repo, añadir step en CI/build que los compile (Cloudflare Pages soporta build script custom; alternativa: compilar localmente y commitear los PDFs)

## Build y deploy

### Local

```
npm install
npm run dev      # arranca en localhost:4321
npm run build    # genera dist/
npm run preview  # sirve dist/ local
```

### Cloudflare Pages

- Repo conectado: `EmanuelAncco/cv-portfolio`
- Branch productivo: `main`
- Branch preview: cualquier otra (auto-preview por PR)
- Framework preset: **Astro**
- Build command: `npm run build`
- Build output: `dist`
- Node version: 22 (LTS)
- Custom domains: `emanuelancco.com` (apex) + `www.emanuelancco.com` (redirige a apex)
- SSL: automático (Cloudflare Universal SSL)

### Compilación de CVs LaTeX

Opción A (recomendada v1): compilar localmente con `tectonic` o `latexmk`, commitear los PDFs en `public/cv/`. Cero overhead en build remoto.

Opción B (v2 si los CVs cambian seguido): añadir build step que use `tectonic` en Cloudflare Pages. Posible pero añade ~30s al build.

## Cosas que NO hace v1 (explícito para evitar scope creep)

- No analytics (no Plausible, no GA, no Cloudflare Web Analytics todavía)
- No formulario de contacto con backend
- No blog
- No CMS
- No i18n EN
- No dark/light toggle (es dark always)
- No animaciones scroll-driven complejas (solo hover sutil + entrada simple)
- No Lenis smooth scroll (browser nativo basta para feel cinematográfico contenido)
- No GSAP en home (si entra, solo en /lab demos)
- No Mermaid diagrams en home (decidir en v2 si se rescatan en /ecosistema)

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Live YOLO del VPS introduce latencia/CORS | Fallback a galería estática de capturas |
| Demo R3F del GNN Puente toma más de lo estimado | Reducir a visualización 2D con D3 si excede 1 día de trabajo |
| Copy editorial nuevo toma tiempo (escribir bien es lento) | Pre-redactar bullets en este spec o en sesión separada antes de tocar Markdown |
| LCP alto por foto grande del hero | `astro:assets` con sizes/srcset + WebP + preload |
| Fonts self-hosted mal configuradas → FOUT | font-display swap + preload de variable woff2 |

## Métricas de éxito

- **Performance:** Lighthouse mobile ≥ 95 en Performance, ≥ 100 en Accessibility, ≥ 100 en Best Practices, ≥ 100 en SEO
- **LCP:** < 2.0s en 4G simulado
- **Bundle:** JS total < 50 KB (homepage). /lab puede subir por R3F.
- **Accesibilidad:** navegación 100% por teclado, contraste AA mínimo, alt text en todas las imágenes
- **SEO:** meta tags + OG image + structured data (Person schema) en home

## Lo que sigue (no es parte del spec, es el siguiente paso)

Implementación dividida en plan de tareas separado (`writing-plans`). Plan tentativo:

1. Scaffold Astro + Tailwind, layout base, fonts self-hosted, CI a Cloudflare Pages
2. Content Collections schema + migrar 12 proyectos a Markdown (sin copy nuevo, copy actual literal)
3. Componentes: Hero, ProjectCard, AxisSection, Footer
4. Páginas: home, archive, cv, sobre, contacto
5. /lab con las 3 demos (puede dividirse en su propio sub-plan)
6. Pase de copy editorial completo (reescritura)
7. Optimización assets, audit Lighthouse, fix accesibilidad
8. Conexión dominio `emanuelancco.com`, redirects, SSL, smoke test
