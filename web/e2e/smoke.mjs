// End-to-end smoke test in a real browser.
//
//   make dev                        # api on :8000, vite on :5173
//   node web/e2e/smoke.mjs          # or: node web/e2e/smoke.mjs http://localhost:5173
//
// Everything else in this repo tests the API. This drives the actual page, because
// the two SSE bugs that made the app look broken to a user -- frames split on "\n\n"
// when the server sends "\r\n\r\n", and a spinner that never stopped -- were both
// invisible to a passing backend suite. A browser is the only place they show up.
//
// Uses the Playwright browsers already installed by `npx playwright install chromium`.
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://localhost:5173";
const results = [];
let failed = 0;

function check(name, ok, detail = "") {
  results.push({ name, ok, detail });
  if (!ok) failed++;
  console.log(`  ${ok ? "pass" : "FAIL"}  ${name}${detail ? `  ${detail}` : ""}`);
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });

// Console errors are a failure on their own: a React crash still renders a shell.
const consoleErrors = [];
page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(String(e)));

try {
  console.log(`\n${BASE}\n`);

  await page.goto(BASE, { waitUntil: "networkidle" });
  check("page loads", (await page.title()).length > 0, await page.title());

  // A session is created on mount; the sample buttons come from /api/samples.
  const sampleButton = page.getByRole("button", { name: /Sales & HR/i });
  await sampleButton.waitFor({ timeout: 15000 });
  check("sample sets are listed", true);

  await sampleButton.click();

  // Five files land as five tables in the left pane.
  await page.getByText(/orders_renamed/i).first().waitFor({ timeout: 60000 });
  const tableNames = ["orders_renamed", "customers", "employees", "payroll_noisy", "weather"];
  const seen = [];
  for (const name of tableNames) {
    if (await page.getByText(new RegExp(name, "i")).first().isVisible()) seen.push(name);
  }
  check("all five sample tables render", seen.length === 5, seen.join(", "));

  // The middle pane is the point of the whole design: joins shown, not assumed.
  const body = await page.locator("body").innerText();
  const hasEvidence = /contain|match|%|confiden/i.test(body);
  check("relationships show their evidence", hasEvidence);

  // Confirm/reject controls must exist -- a human overruling the graph is the design.
  // Confirm is hidden on an edge that is already confirmed, so Reject is the control
  // present on a freshly discovered graph. Its existence is the claim being checked:
  // a human can overrule the graph.
  const overrulable = await page.getByRole("button", { name: /reject/i }).count();
  check("relationships can be overruled by a human", overrulable > 0, `${overrulable} controls`);

  // Ask a real question and wait for the answer, not just for the request to finish.
  const box = page.getByRole("textbox").last();
  await box.fill("What is the total order amount per region?");
  await box.press("Enter");

  // waitForFunction takes (fn, arg, options) -- passing options second silently made
  // this the default 30 s, which is fine locally and far too short against a free-tier
  // hosted model.
  await page.waitForFunction(
    () => /\$?\s?1?0?9[,.]?3\d\d/.test(document.body.innerText) ||
          /South/.test(document.body.innerText),
    null,
    { timeout: 300000 },
  );
  const answer = await page.locator("body").innerText();
  check("a prose answer appears", /South|North|region/i.test(answer));
  check("the numbers are in the answer", /109|79|98|102/.test(answer));

  // The chart is an SVG from recharts; a table-only fallback would have no <svg>.
  const svgs = await page.locator("svg").count();
  check("a chart is rendered", svgs > 0, `${svgs} svg nodes`);

  // The trace is the auditability claim. It is behind a disclosure that closes once
  // the answer lands -- deliberate, so the SQL is one click rather than always in the
  // way -- so the test opens it rather than asserting on the collapsed text.
  for (const d of await page.locator("details").all()) {
    await d.evaluate((el) => el.setAttribute("open", ""));
  }
  const expanded = await page.locator("body").innerText();
  check("the generated SQL is shown", /select/i.test(expanded));
  check("the trace lists its steps", /route|plan|execute/i.test(expanded));
  check("the rows behind the answer are available", /rows behind this answer/i.test(expanded));

  await page.screenshot({ path: "web/e2e/smoke.png", fullPage: false });
  console.log("\n  screenshot -> web/e2e/smoke.png");

  // Everything above is the happy path, so nothing should have gone to the console.
  // Checked here rather than at the end, because the recovery test below provokes a
  // 404 on purpose and would otherwise fail this.
  check("no console errors", consoleErrors.length === 0, consoleErrors.slice(0, 2).join(" | "));

  // Session expiry must recover rather than dead-end. Delete the session out from
  // under the page, then drive it again: the app should start a fresh one and say so.
  const sessionId = await page.evaluate(async () => {
    const r = await fetch("/api/sessions", { method: "POST" });
    return (await r.json()).session_id;
  });
  await page.evaluate(async (id) => {
    await fetch(`/api/sessions/${id}`, { method: "DELETE" });
  }, sessionId);
  const gone = await page.evaluate(async (id) => {
    const r = await fetch(`/api/sessions/${id}/schema`);
    return r.status;
  }, sessionId);
  check("a deleted session is a clean 404", gone === 404, `status ${gone}`);

  // And the page recovers instead of dead-ending. The page's own session id is not
  // exposed, so the server's answer is forced instead: the next relationship edit
  // comes back SESSION_NOT_FOUND, which is exactly what an evicted session looks
  // like to this client. It must start a fresh session rather than show a dead screen.
  await page.route("**/api/sessions/*/relationships/*", (route) =>
    route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ error: { code: "SESSION_NOT_FOUND", message: "gone" } }),
    }),
  );
  await page.getByRole("button", { name: /reject/i }).first().click();
  await page.waitForTimeout(2500);
  await page.unroute("**/api/sessions/*/relationships/*");

  const recovered = await page.locator("body").innerText();
  check(
    "an expired session is explained, not fatal",
    /expired|new one was started|add your files again/i.test(recovered),
    recovered.split("\n").find((l) => /expired/i.test(l))?.slice(0, 60) ?? "",
  );
  // The ask box is correctly disabled with no tables loaded, so "usable" means the
  // user can start again: the sample sets come back, which is what the notice tells
  // them to do.
  check(
    "the user can start over after recovery",
    await page.getByRole("button", { name: /Sales & HR/i }).isVisible(),
  );
} catch (error) {
  check("ran to completion", false, String(error).slice(0, 300));
} finally {
  await browser.close();
}

console.log(`\n${results.length - failed}/${results.length} checks passed\n`);
process.exit(failed ? 1 : 0);
