// Optional: npm install --no-save --package-lock=false playwright
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('playwright');
(async () => {
  const root = path.resolve(__dirname, '..');
  const browser = await chromium.launch({ headless: true,
    ...(process.env.NETOPS_FAULT_BROWSER ? { executablePath: process.env.NETOPS_FAULT_BROWSER } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 1160 } });
    const errors = [];
    let requests = 0;
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (/^https?:/.test(r.url())) requests++; });
    await page.goto(pathToFileURL(path.join(root, 'docs/ambiguous.html')).href);
    await page.getByRole('heading', { name: 'These links fit the same observations.' }).waitFor();
    await page.getByRole('heading', { name: 'probe-core', exact: true }).waitFor();
    await page.screenshot({ path: path.join(root, 'docs/assets/report.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    await page.screenshot({ path: path.join(root, 'docs/assets/report-mobile.png'), fullPage: true });
    await page.getByText('Model assumptions and parameters', { exact: true }).click();
    if (!(await page.getByText('At most one failed link, known static paths', { exact: false }).isVisible())) throw new Error('Details not visible');
    await page.setViewportSize({ width: 1200, height: 1160 });
    await page.goto(pathToFileURL(path.join(root, 'docs/after-probe.html')).href);
    await page.getByRole('heading', { name: 'A leading candidate to investigate.' }).waitFor();
    await page.screenshot({ path: path.join(root, 'docs/assets/after-probe.png'), fullPage: true });
    if (errors.length || requests || mobileOverflow) throw new Error(JSON.stringify({ errors, requests, mobileOverflow }));
    const result = { browser: browser.version(), viewports: ['1200x1160','390x844'], externalRequests: requests,
      pageErrors: errors, mobileOverflow, beforeAndAfterStates: 'passed', nativeDetails: 'passed' };
    fs.writeFileSync(path.join(root, 'benchmarks/results/ui.json'), JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode=1; });
