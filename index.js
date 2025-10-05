import 'dotenv/config';
import fs from 'node:fs/promises';
import path from 'node:path';
import { Bot } from 'grammy';
import { openai } from '@ai-sdk/openai';
import { generateText, tool } from 'ai';
import { z } from 'zod';
import dedent from "dedent";
import { google } from 'googleapis';


// ----- Config -----
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const DOC_PATH = path.resolve(process.cwd(), 'doc.txt');
const ADMIN_CHAT_ID = process.env.ADMIN_CHAT_ID ? Number(process.env.ADMIN_CHAT_ID) : undefined;
const BOOKINGS_PATH = path.resolve(process.cwd(), 'bookings.json');
const GOOGLE_SHEETS_SPREADSHEET_ID = process.env.GOOGLE_SHEETS_SPREADSHEET_ID;
const GOOGLE_SHEETS_SHEET_NAME = process.env.GOOGLE_SHEETS_SHEET_NAME || 'bookings';
const GOOGLE_CREDENTIALS_JSON = process.env.GOOGLE_CREDENTIALS_JSON;
const GOOGLE_SERVICE_ACCOUNT_EMAIL = process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL;
const GOOGLE_PRIVATE_KEY = process.env.GOOGLE_PRIVATE_KEY;

if (!TELEGRAM_BOT_TOKEN) {
  console.error('Missing TELEGRAM_BOT_TOKEN in environment.');
  process.exit(1);
}
if (!OPENAI_API_KEY) {
  console.error('Missing OPENAI_API_KEY in environment.');
  process.exit(1);
}

// ----- Knowledge Base in System Prompt -----
const DOC_TEXT = await fs.readFile(DOC_PATH, 'utf8');

const TIME_ZONE = process.env.TIME_ZONE || 'Asia/Omsk';

function getNowString() {
  const now = new Date();
  try {
    return new Intl.DateTimeFormat('ru-RU', {
      timeZone: TIME_ZONE,
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(now);
  } catch {
    return now.toISOString();
  }
}

function getNowWeekdayRu() {
  const now = new Date();
  try {
    return new Intl.DateTimeFormat('ru-RU', {
      timeZone: TIME_ZONE,
      weekday: 'long',
    }).format(now);
  } catch {
    const names = ['воскресенье','понедельник','вторник','среда','четверг','пятница','суббота'];
    return names[now.getUTCDay()];
  }
}

function getTzParts(date, timeZone) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).formatToParts(date).reduce((acc, p) => {
    acc[p.type] = p.value;
    return acc;
  }, {});
  return {
    year: Number(parts.year),
    month: Number(parts.month),
    day: Number(parts.day),
    hour: Number(parts.hour),
    minute: Number(parts.minute),
    second: Number(parts.second),
  };
}

function getTzOffsetMinutes(date, timeZone) {
  const { year, month, day, hour, minute, second } = getTzParts(date, timeZone);
  const asUTC = Date.UTC(year, month - 1, day, hour, minute, second, 0);
  // Offset = local_time(UTC) - actual_utc
  return Math.round((asUTC - date.getTime()) / 60000);
}

function formatLocalRFC3339(date, timeZone) {
  const { year, month, day, hour, minute, second } = getTzParts(date, timeZone);
  const pad = (n) => String(n).padStart(2, '0');
  const offMin = getTzOffsetMinutes(date, timeZone);
  const sign = offMin >= 0 ? '+' : '-';
  const abs = Math.abs(offMin);
  const offH = Math.floor(abs / 60);
  const offM = abs % 60;
  const offset = `${sign}${pad(offH)}:${pad(offM)}`;
  return `${year}-${pad(month)}-${pad(day)}T${pad(hour)}:${pad(minute)}:${pad(second)}${offset}`;
}

function formatLocalPlain(date, timeZone) {
  const { year, month, day, hour, minute, second } = getTzParts(date, timeZone);
  const pad = (n) => String(n).padStart(2, '0');
  return `${year}-${pad(month)}-${pad(day)}T${pad(hour)}:${pad(minute)}:${pad(second)}`;
}

function addDaysLocalParts(parts, days) {
  const guess = new Date(zonedDateTimeToUTCISO(parts.year, parts.month, parts.day, parts.hour, parts.minute, TIME_ZONE));
  const next = new Date(guess.getTime() + days * 24 * 60 * 60 * 1000);
  const p = getTzParts(next, TIME_ZONE);
  return { year: p.year, month: p.month, day: p.day, hour: parts.hour, minute: parts.minute };
}

function normalizeWhenLocalPair(input) {
  const src = String(input || '').trim();
  const lower = src.toLowerCase();
  const now = new Date();
  const nowParts = getTzParts(now, TIME_ZONE);

  const extractTime = (source) => {
    let h = 12, m = 0, found = false;
    const m1 = source.match(/\bв\s*(\d{1,2})(?::(\d{2}))?\b/);
    if (m1) { h = Math.min(23, Math.max(0, parseInt(m1[1], 10))); m = m1[2] ? Math.min(59, Math.max(0, parseInt(m1[2], 10))) : 0; found = true; }
    else {
      const m2 = source.match(/\b(\d{1,2}):(\d{2})\b/);
      if (m2) { h = Math.min(23, Math.max(0, parseInt(m2[1], 10))); m = Math.min(59, Math.max(0, parseInt(m2[2], 10))); found = true; }
    }
    return { found, h, m };
  };

  const build = (y, mo, d, h, mi) => {
    const utcISO = zonedDateTimeToUTCISO(y, mo, d, h, mi, TIME_ZONE);
    const localPlain = `${y}-${String(mo).padStart(2,'0')}-${String(d).padStart(2,'0')}T${String(h).padStart(2,'0')}:${String(mi).padStart(2,'0')}:00`;
    return { utcISO, localPlain };
  };

  // Weekdays
  const weekdayDefs = [
    { re: /\bпонедельник\b|\bв\s+понедельник\b/, idx: 1 },
    { re: /\bвторник\b|\bво\s+вторник\b/, idx: 2 },
    { re: /\bсред[ауы]\b|\bв\s+сред[ау]\b/, idx: 3 },
    { re: /\bчетверг\b|\bв\s+четверг\b/, idx: 4 },
    { re: /\bпятниц[ауы]\b|\bв\s+пятниц[ау]\b/, idx: 5 },
    { re: /\bсуббот[ауы]\b|\bв\s+суббот[уa]\b/, idx: 6 },
    { re: /\bвоскресенье\b|\bв\s+воскресенье\b/, idx: 0 },
  ];
  const nowDOW = new Date(zonedDateTimeToUTCISO(nowParts.year, nowParts.month, nowParts.day, nowParts.hour, nowParts.minute, TIME_ZONE)).getDay();
  for (const def of weekdayDefs) {
    if (def.re.test(lower)) {
      const t = extractTime(lower);
      const h = t.found ? t.h : 12;
      const m = t.found ? t.m : 0;
      let delta = def.idx - nowDOW; if (delta < 0) delta += 7;
      let target = addDaysLocalParts({ ...nowParts, hour: h, minute: m }, delta);
      // if it's same day and passed: add 7 days
      const targetDate = new Date(zonedDateTimeToUTCISO(target.year, target.month, target.day, h, m, TIME_ZONE));
      const nowDate = new Date(zonedDateTimeToUTCISO(nowParts.year, nowParts.month, nowParts.day, nowParts.hour, nowParts.minute, TIME_ZONE));
      if (targetDate.getTime() <= nowDate.getTime()) {
        const plus = addDaysLocalParts(target, 7);
        target = { ...plus, hour: h, minute: m };
      }
      return build(target.year, target.month, target.day, h, m);
    }
  }

  // Relative words
  if (/(послезавтра)/.test(lower)) {
    const t = extractTime(lower);
    const h = t.found ? t.h : 12;
    const m = t.found ? t.m : 0;
    const target = addDaysLocalParts({ ...nowParts, hour: h, minute: m }, 2);
    return build(target.year, target.month, target.day, h, m);
  }
  if (/(завтра)/.test(lower)) {
    const t = extractTime(lower);
    const h = t.found ? t.h : 12;
    const m = t.found ? t.m : 0;
    const target = addDaysLocalParts({ ...nowParts, hour: h, minute: m }, 1);
    return build(target.year, target.month, target.day, h, m);
  }
  if (/(сегодня)/.test(lower)) {
    const t = extractTime(lower);
    const h = t.found ? t.h : 12;
    const m = t.found ? t.m : 0;
    return build(nowParts.year, nowParts.month, nowParts.day, h, m);
  }

  // DD.MM(.YYYY)? with time
  const dm = lower.match(/\b(\d{1,2})[.](\d{1,2})(?:[.](\d{4}))?\b/);
  if (dm) {
    const day = parseInt(dm[1], 10);
    const month = parseInt(dm[2], 10);
    const year = dm[3] ? parseInt(dm[3], 10) : nowParts.year;
    const remainder = lower.replace(dm[0], ' ');
    const t = extractTime(remainder);
    const h = t.found ? t.h : 12;
    const m = t.found ? t.m : 0;
    return build(year, month, day, h, m);
  }

  // Fallback: if parseable, project into TZ local
  const parsed = new Date(src);
  if (!isNaN(parsed.getTime())) {
    const p = getTzParts(parsed, TIME_ZONE);
    return build(p.year, p.month, p.day, p.hour, p.minute);
  }
  return { utcISO: src, localPlain: src };
}

function parseLocalPlainToDate(localPlain) {
  const m = localPlain.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})$/);
  if (!m) return new Date(localPlain);
  const [_, y, mo, d, h, mi, s] = m;
  const iso = zonedDateTimeToUTCISO(parseInt(y,10), parseInt(mo,10), parseInt(d,10), parseInt(h,10), parseInt(mi,10), TIME_ZONE);
  return new Date(iso);
}

// Convert a local civil time in a time zone to UTC ISO, DST-aware
function zonedDateTimeToUTCISO(year, month, day, hour, minute, timeZone) {
  // Initial guess as UTC
  let guessMs = Date.UTC(year, month - 1, day, hour, minute, 0, 0);
  // Iteratively refine offset (handles DST boundaries)
  for (let i = 0; i < 2; i++) {
    const offsetMin = getTzOffsetMinutes(new Date(guessMs), timeZone);
    const targetMs = Date.UTC(year, month - 1, day, hour, minute, 0, 0) - offsetMin * 60000;
    if (Math.abs(targetMs - guessMs) < 60000) { // within 1 minute
      guessMs = targetMs;
      break;
    }
    guessMs = targetMs;
  }
  return new Date(guessMs).toISOString();
}

function isLateNightLocal(timeZone) {
  const now = new Date();
  const p = getTzParts(now, timeZone);
  const h = p.hour;
  return h >= 22 || h < 4;
}

function hasExplicitCalendarDate(input) {
  const s = String(input || '').toLowerCase();
  if (/\b\d{4}-\d{2}-\d{2}\b/.test(s)) return true; // ISO date
  if (/\b\d{1,2}[.]\d{1,2}([.]\d{2,4})?\b/.test(s)) return true; // DD.MM(.YYYY)
  return false;
}

function containsRelativeOrWeekday(input) {
  const s = String(input || '').toLowerCase();
  if (/(сегодня|завтра|послезавтра)/.test(s)) return true;
  if (/(понедельник|вторник|среда|четверг|пятница|суббота|воскресенье)/.test(s)) return true;
  return false;
}

function formatDateDDMMYYYY(date) {
  const d = parseLocalPlainToDate(date);
  if (isNaN(d.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  const parts = getTzParts(d, TIME_ZONE);
  return `${pad(parts.day)}.${pad(parts.month)}.${parts.year}`;
}

function buildSystemPrompt() {
  console.log('buildSystemPrompt', getNowString());
  return [
    'Ты — ИИ‑администратор фитнес‑клуба IronPulse. ',
    'Отвечай кратко, точно и дружелюбно, только на русском. ',
    'Используй ТОЛЬКО информацию из документа ниже. ',
    'Если ответа нет в документе, так и скажи и предложи связаться: +7 (495) 555-55-55, info@ironpulse.ru. ',
    'Форматируй ключевые списки пунктами. ',
    'Не используй Markdown или HTML и не добавляй преамбулы. Можешь использовать эмиджи.',
    `Текущая дата и время (${TIME_ZONE}): ${getNowString()}. Сегодня: ${getNowWeekdayRu()}. Для выражений "сегодня/завтра/в пятницу" всегда считай от текущей даты. Не используй устаревший 2023 год. `,
    'Если пользователь хочет записаться на пробное занятие или соглашается на предложение, последовательно спроси недостающие данные: имя, телефон, дата визита. Время НЕ спрашивай. Если время не указано — используй 12:00 локального времени. После того, как всё собрано, вызови инструмент book_trial. Не выдумывай данные. ',
    'Если пользователь хочет перенести пробную запись, уточни новую дату (время — по желанию пользователя; если не указал, используй 12:00) и вызови инструмент reschedule_trial.\n\n',
    'Ночное уточнение: если пользователь пишет поздно вечером или ночью (примерно 22:00–04:00 местного времени) и использует относительные выражения (сегодня/завтра/послезавтра/в [день недели]) без календарной даты, ОБЯЗАТЕЛЬНО уточни календарную дату в формате ДД.ММ.ГГГГ перед бронированием/переносом. Например: «Уточню, вы имеете в виду 05.10.2025?»\n\n',
    'Если пользователь хочет отменить пробную запись, сначала получи ближайшую будущую запись через get_upcoming_trial и покажи её пользователю. Затем спроси подтверждение отмены. При подтверждении вызови cancel_trial с точным when. Отменять можно только будущие записи. После отмены можно записаться снова. Для поиска и управления записями используй userId, телефон не запрашивай.\n\n',
    'Документ:\n',
    DOC_TEXT,
  ].join('');
}

const bot = new Bot(TELEGRAM_BOT_TOKEN);

// ----- Persistence for bookings -----
const SHEET_HEADER = ['userId','name','phone','phoneNormalized','when','createdAt','updatedAt','status','canceledAt'];

let cachedSheetsClient = null; // { sheets, spreadsheetId, sheetName }

function parseGoogleCredentialsFromEnv() {
  // Prefer GOOGLE_CREDENTIALS_JSON if present (can be raw JSON or base64-encoded JSON)
  if (GOOGLE_CREDENTIALS_JSON) {
    let raw = String(GOOGLE_CREDENTIALS_JSON);
    try {
      const trimmed = raw.trim();
      if (!trimmed.startsWith('{')) {
        // Try base64 decode
        try {
          raw = Buffer.from(trimmed, 'base64').toString('utf8');
        } catch {}
      }
      const parsed = JSON.parse(raw);
      const client_email = parsed.client_email;
      let private_key = parsed.private_key;
      if (typeof private_key === 'string') {
        // Normalize escaped newlines and CRLF
        private_key = private_key.replace(/\\n/g, '\n').replace(/\r\n/g, '\n');
      }
      if (client_email && private_key) {
        return { client_email, private_key };
      }
    } catch {}
  }
  // Fallback to pair of vars
  if (GOOGLE_SERVICE_ACCOUNT_EMAIL && GOOGLE_PRIVATE_KEY) {
    return {
      client_email: GOOGLE_SERVICE_ACCOUNT_EMAIL,
      private_key: GOOGLE_PRIVATE_KEY.replace(/\\n/g, '\n'),
    };
  }
  return null;
}

async function getSheetsClient() {
  try {
    if (!GOOGLE_SHEETS_SPREADSHEET_ID) return null;
    if (cachedSheetsClient) return cachedSheetsClient;
    const credentials = parseGoogleCredentialsFromEnv();
    if (!credentials) return null;
    const auth = new google.auth.GoogleAuth({
      credentials,
      scopes: ['https://www.googleapis.com/auth/spreadsheets'],
    });
    const client = await auth.getClient();
    const sheets = google.sheets({ version: 'v4', auth: client });
    cachedSheetsClient = { sheets, spreadsheetId: GOOGLE_SHEETS_SPREADSHEET_ID, sheetName: GOOGLE_SHEETS_SHEET_NAME };
    return cachedSheetsClient;
  } catch (e) {
    console.error('Sheets auth failed:', e);
    return null;
  }
}

async function ensureSheetHeader() {
  const cli = await getSheetsClient();
  if (!cli) return false;
  const { sheets, spreadsheetId, sheetName } = cli;
  try {
    const resp = await sheets.spreadsheets.values.get({
      spreadsheetId,
      range: `${sheetName}!A1:Z1`,
      majorDimension: 'ROWS',
    });
    const values = resp.data.values || [];
    const firstRow = values[0] || [];
    const equal = SHEET_HEADER.length === firstRow.length && SHEET_HEADER.every((h, i) => firstRow[i] === h);
    if (!firstRow.length) {
      await sheets.spreadsheets.values.update({
        spreadsheetId,
        range: `${sheetName}!A1`,
        valueInputOption: 'RAW',
        requestBody: { values: [SHEET_HEADER] },
      });
    } else if (!equal) {
      // Overwrite header to our canonical set
      await sheets.spreadsheets.values.update({
        spreadsheetId,
        range: `${sheetName}!A1`,
        valueInputOption: 'RAW',
        requestBody: { values: [SHEET_HEADER] },
      });
    }
    return true;
  } catch (e) {
    console.error('ensureSheetHeader failed:', e);
    return false;
  }
}

async function sheetLoadBookings() {
  try {
    const cli = await getSheetsClient();
    if (!cli) return null;
    const { sheets, spreadsheetId, sheetName } = cli;
    await ensureSheetHeader();
    const resp = await sheets.spreadsheets.values.get({
      spreadsheetId,
      range: `${sheetName}!A1:Z`,
      majorDimension: 'ROWS',
    });
    const rows = resp.data.values || [];
    if (!rows.length) return [];
    const header = rows[0];
    const list = [];
    for (let i = 1; i < rows.length; i++) {
      const row = rows[i] || [];
      const obj = {};
      for (let c = 0; c < header.length; c++) {
        obj[header[c]] = row[c] ?? '';
      }
      if (obj.userId !== undefined && obj.userId !== null && obj.userId !== '') {
        const n = Number(obj.userId);
        if (!Number.isNaN(n)) obj.userId = n;
      }
      list.push(obj);
    }
    return list;
  } catch (e) {
    console.error('sheetLoadBookings failed:', e);
    return null;
  }
}

async function sheetAppendBooking(record) {
  try {
    const cli = await getSheetsClient();
    if (!cli) return false;
    const { sheets, spreadsheetId, sheetName } = cli;
    const ok = await ensureSheetHeader();
    if (!ok) return false;
    const row = SHEET_HEADER.map((k) => {
      const v = record[k];
      return v === undefined || v === null ? '' : String(v);
    });
    await sheets.spreadsheets.values.append({
      spreadsheetId,
      range: `${sheetName}!A1`,
      valueInputOption: 'RAW',
      insertDataOption: 'INSERT_ROWS',
      requestBody: { values: [row] },
    });
    return true;
  } catch (e) {
    console.error('sheetAppendBooking failed:', e);
    return false;
  }
}

async function sheetSaveBookings(list) {
  try {
    const cli = await getSheetsClient();
    if (!cli) return false;
    const { sheets, spreadsheetId, sheetName } = cli;
    const ok = await ensureSheetHeader();
    if (!ok) return false;
    const rows = list.map((rec) => SHEET_HEADER.map((k) => {
      const v = rec[k];
      return v === undefined || v === null ? '' : String(v);
    }));
    // Clear existing data below header
    await sheets.spreadsheets.values.clear({
      spreadsheetId,
      range: `${sheetName}!A2:Z`,
    });
    if (rows.length) {
      await sheets.spreadsheets.values.update({
        spreadsheetId,
        range: `${sheetName}!A2`,
        valueInputOption: 'RAW',
        requestBody: { values: rows },
      });
    }
    return true;
  } catch (e) {
    console.error('sheetSaveBookings failed:', e);
    return false;
  }
}

async function appendBooking(record) {
  // Try Google Sheets first
  const sheetOk = await sheetAppendBooking(record);
  if (sheetOk) return;
  // Fallback to file
  try {
    let list = [];
    try {
      const raw = await fs.readFile(BOOKINGS_PATH, 'utf8');
      list = JSON.parse(raw);
      if (!Array.isArray(list)) list = [];
    } catch {}
    list.push(record);
    await fs.writeFile(BOOKINGS_PATH, JSON.stringify(list, null, 2), 'utf8');
  } catch (err) {
    console.error('Failed to persist booking (file fallback):', err);
  }
}

function toolResultsToText(toolResults) {
  if (!toolResults || !toolResults.length) return '';
  try {
    return toolResults
      .map((r) => {
        const v = (r && (r.output ?? r.result));
        if (typeof v === 'string') return v;
        try { return JSON.stringify(v); } catch { return String(v); }
      })
      .filter(Boolean)
      .join('\n');
  } catch {
    return '';
  }
}

async function loadBookings() {
  // Try Google Sheets first
  const sheetList = await sheetLoadBookings();
  if (Array.isArray(sheetList)) return sheetList;
  // Fallback to file
  try {
    const raw = await fs.readFile(BOOKINGS_PATH, 'utf8');
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

function normalizePhone(phone) {
  if (!phone) return '';
  const digits = String(phone).replace(/\D+/g, '');
  // Normalize Russian numbers: allow leading 8 -> 7
  if (digits.length === 11 && (digits.startsWith('8') || digits.startsWith('7'))) {
    return '7' + digits.slice(1);
  }
  if (digits.length === 10) {
    return '7' + digits; // assume Russia without country code
  }
  return digits;
}

async function notifyAdmin(record) {
  if (!ADMIN_CHAT_ID) return;
  const lines = [
    'Новая запись на пробное занятие:',
    `Имя: ${record.name}`,
    `Телефон: ${record.phone}`,
    `Когда: ${record.when}`,
    `Telegram id: ${record.userId}`,
  ];
  try {
    await bot.api.sendMessage(ADMIN_CHAT_ID, lines.join('\n'));
  } catch (e) {
    console.error('Admin notify failed:', e);
  }
}

async function notifyAdminReschedule(record, oldWhen) {
  if (!ADMIN_CHAT_ID) return;
  const lines = [
    'Перенос пробной тренировки:',
    `Имя: ${record.name}`,
    `Телефон: ${record.phone}`,
    `Было: ${oldWhen}`,
    `Стало: ${record.when}`,
    `Telegram id: ${record.userId}`,
  ];
  try {
    await bot.api.sendMessage(ADMIN_CHAT_ID, lines.join('\n'));
  } catch (e) {
    console.error('Admin notify (reschedule) failed:', e);
  }
}

async function saveBookings(list) {
  // Try Google Sheets first
  const sheetOk = await sheetSaveBookings(list);
  if (sheetOk) return;
  // Fallback to file
  try {
    await fs.writeFile(BOOKINGS_PATH, JSON.stringify(list, null, 2), 'utf8');
  } catch (e) {
    console.error('Failed to save bookings (file fallback):', e);
  }
}

function parseWhenToDate(when) {
  try {
    const d = new Date(when);
    if (isNaN(d.getTime())) return null;
    return d;
  } catch {
    return null;
  }
}

function isFutureDate(dt) {
  if (!dt) return false;
  const now = new Date();
  return dt.getTime() > now.getTime();
}

// ----- Agent tool factory -----
function buildTools(telegramUserId) {
  return {
    get_upcoming_trial: tool({
      description: 'Вернёт ближайшую будущую пробную запись пользователя по userId (для показа перед отменой/переносом).',
      inputSchema: z.object({}),
      execute: async () => {
        const list = await loadBookings();
        const candidates = list
          .filter((b) => b.userId === telegramUserId && b.status !== 'canceled')
          .map((b) => ({ b, dt: parseWhenToDate(b.when) }))
          .filter(({ dt }) => isFutureDate(dt));
        if (!candidates.length) return 'Будущих записей не найдено.';
        candidates.sort((a, b) => a.dt.getTime() - b.dt.getTime());
        const target = candidates[0].b;
        let human = target.when;
        try {
          const d = parseLocalPlainToDate(target.when);
          if (!isNaN(d.getTime())) {
            human = d.toLocaleDateString('ru-RU', { 
              timeZone: TIME_ZONE,
              year: 'numeric', 
              month: '2-digit', 
              day: '2-digit' 
            });
          }
        } catch {}
        return `Ваша ближайшая запись: ${human}. Хотите отменить?`;
      },
    }),
    book_trial: tool({
      description: 'Забронировать пробное занятие. Вызови после того, как узнал имя, телефон и ДАТУ визита (время не требуется, по умолчанию 12:00). При обращениях поздно вечером/ночью и относительных формулировках сначала уточни календарную дату.',
      inputSchema: z.object({
        name: z.string().min(1),
        phone: z.string().min(5),
        when: z.string().min(1),
      }),
      execute: async ({ name, phone, when }) => {
        // Late-night clarification for relative expressions without explicit date
        if (containsRelativeOrWeekday(when) && !hasExplicitCalendarDate(when) && isLateNightLocal(TIME_ZONE)) {
          const { localPlain } = normalizeWhenLocalPair(when);
          const ddmmyyyy = formatDateDDMMYYYY(localPlain);
          if (ddmmyyyy) {
            return `Уточню: вы имеете в виду ${ddmmyyyy}? Ответьте полной датой в формате ДД.ММ.ГГГГ.`;
          }
          return 'Уточните, пожалуйста, точную дату в формате ДД.ММ.ГГГГ.';
        }
        const phoneNormalized = normalizePhone(phone);
        const existing = await loadBookings();
        const alreadyHas = existing.some((b) => {
          const bPhone = normalizePhone(b.phoneNormalized || b.phone);
          const isSameUserOrPhone = b.userId === telegramUserId || (phoneNormalized && bPhone && bPhone === phoneNormalized);
          const isCanceled = b.status === 'canceled';
          return isSameUserOrPhone && !isCanceled;
        });
        if (alreadyHas) {
          return [
            'Похоже, у вас уже есть запись на пробную тренировку. ',
            'Если нужно изменить дату/время — напишите, я помогу.'
          ].join('');
        }
        const { utcISO, localPlain } = normalizeWhenLocalPair(when);
        const record = {
          userId: telegramUserId,
          name,
          phone,
          phoneNormalized,
          when: localPlain,
          createdAt: new Date().toISOString(),
        };
        console.log('Booking record:', record);
        await appendBooking(record);
        await notifyAdmin(record);
        let human = localPlain;
        try {
          const d = parseLocalPlainToDate(localPlain);
          if (!isNaN(d.getTime())) {
            human = d.toLocaleDateString('ru-RU', { 
              timeZone: TIME_ZONE,
              year: 'numeric', 
              month: '2-digit', 
              day: '2-digit' 
            });
          }
        } catch {}
        return `Запись создана: ${name}, ${phone}, ${human}. Мы свяжемся для подтверждения.`;
      },
    }),
    reschedule_trial: tool({
      description: 'Перенести (перезаписать) существующую пробную запись на новую дату. Время не обязательно (по умолчанию 12:00). При обращениях поздно вечером/ночью и относительных формулировках сначала уточни календарную дату.',
      inputSchema: z.object({ newWhen: z.string().min(1) }),
      execute: async ({ newWhen }) => {
        const list = await loadBookings();
        // Найти последнюю запись пользователя
        const candidates = list.filter((b) => b.userId === telegramUserId && b.status !== 'canceled');
        if (!candidates.length) {
          return [
            'Не нашёл существующую пробную запись для переноса. ',
            'Если у вас её ещё нет — могу оформить новую.'
          ].join('');
        }
        // Самая свежая по createdAt
        const target = candidates.reduce((acc, cur) => {
          const tAcc = new Date(acc.createdAt || 0).getTime();
          const tCur = new Date(cur.createdAt || 0).getTime();
          return tCur >= tAcc ? cur : acc;
        });
        const oldWhen = target.when;
        // Late-night clarification for relative expressions without explicit date
        if (containsRelativeOrWeekday(newWhen) && !hasExplicitCalendarDate(newWhen) && isLateNightLocal(TIME_ZONE)) {
          const { localPlain } = normalizeWhenLocalPair(newWhen);
          const ddmmyyyy = formatDateDDMMYYYY(localPlain);
          if (ddmmyyyy) {
            return `Уточню: вы имеете в виду ${ddmmyyyy}? Ответьте полной датой в формате ДД.ММ.ГГГГ.`;
          }
          return 'Уточните, пожалуйста, точную дату в формате ДД.ММ.ГГГГ.';
        }
        const { utcISO, localPlain } = normalizeWhenLocalPair(newWhen);
        target.when = localPlain;
        target.updatedAt = new Date().toISOString();
        await saveBookings(list);
        await notifyAdminReschedule(target, oldWhen);
        let human = localPlain;
        try {
          const d = parseLocalPlainToDate(localPlain);
          if (!isNaN(d.getTime())) {
            human = d.toLocaleDateString('ru-RU', { 
              timeZone: TIME_ZONE,
              year: 'numeric', 
              month: '2-digit', 
              day: '2-digit' 
            });
          }
        } catch {}
        let oldHuman = oldWhen;
        try {
          const oldD = parseLocalPlainToDate(oldWhen);
          if (!isNaN(oldD.getTime())) {
            oldHuman = oldD.toLocaleDateString('ru-RU', { 
              timeZone: TIME_ZONE,
              year: 'numeric', 
              month: '2-digit', 
              day: '2-digit' 
            });
          }
        } catch {}
        return `Перенос выполнен: было «${oldHuman}», стало «${human}».`;
      },
    }),
    cancel_trial: tool({
      description: 'Отменить пробную запись по userId.',
      inputSchema: z.object({ when: z.string().min(1) }),
      execute: async ({ when }) => {
        const list = await loadBookings();
        // Кандидаты: по пользователю и не отменённые
        let candidates = list.filter((b) => b.userId === telegramUserId && b.status !== 'canceled');
        if (!candidates.length) return 'Не нашёл активных записей для отмены.';

        // when обязателен (подтверждение)
        const { utcISO, localPlain } = normalizeWhenLocalPair(when);
        let target = candidates.find((b) => String(b.when).trim() === localPlain.trim()) || null;
        if (!target) {
          const targetDt = parseLocalPlainToDate(localPlain);
          if (targetDt) {
            candidates = candidates.filter((b) => !!parseLocalPlainToDate(b.when));
            target = candidates.find((b) => {
              const bd = parseLocalPlainToDate(b.when);
              return bd && Math.abs(bd.getTime() - targetDt.getTime()) < 5 * 60 * 1000;
            }) || null;
          }
        }
        if (!target) return 'Не удалось найти запись с указанной датой/временем. Уточните, пожалуйста.';

        // Если when не задан — выбрать ближайшую будущую
        // Проверка, что запись в будущем
        const dt = parseWhenToDate(target.when);
        if (!isFutureDate(dt)) {
          return 'Эту запись уже нельзя отменить (кажется, время прошло).';
        }

        // Пометить как отменённую
        target.status = 'canceled';
        target.canceledAt = new Date().toISOString();
        await saveBookings(list);
        // Сообщить админу
        try {
          if (ADMIN_CHAT_ID) {
            await bot.api.sendMessage(
              ADMIN_CHAT_ID,
              [
                'Отмена пробной тренировки:',
                `Имя: ${target.name}`,
                `Телефон: ${target.phone}`,
                `Когда: ${target.when}`,
                `Telegram id: ${target.userId}`,
              ].join('\n')
            );
          }
        } catch (e) {
          console.error('Admin notify (cancel) failed:', e);
        }

        let human = target.when;
        try {
          const d = parseLocalPlainToDate(target.when);
          if (!isNaN(d.getTime())) {
            human = d.toLocaleDateString('ru-RU', { 
              timeZone: TIME_ZONE,
              year: 'numeric', 
              month: '2-digit', 
              day: '2-digit' 
            });
          }
        } catch {}
        return `Запись отменена: ${human}. Могу помочь выбрать новую дату.`;
      },
    }),
  };
}

// ----- Simple per-chat memory -----
const chatHistory = new Map(); // chatId -> [{ role, content }]
function getHistory(chatId) {
  if (!chatHistory.has(chatId)) chatHistory.set(chatId, []);
  return chatHistory.get(chatId);
}

bot.command('start', async (ctx) => {
  await ctx.reply(
    dedent`Привет!

    Я администратор IronPulse. Задайте вопрос: цены, расписание, услуги, контакты.
    Или напишите мне свой номер телефона, и я свяжусь с вами.`
  );
});

bot.on('message:contact', async (ctx) => {
  const chatId = ctx.chat.id;
  const messages = getHistory(chatId);
  const phone = ctx.message.contact?.phone_number;
  if (!phone) return;
  messages.push({ role: 'user', content: `Мой номер телефона: ${phone}` });
  try {
    await ctx.api.sendChatAction(ctx.chat.id, 'typing');
    const tools = buildTools(ctx.from?.id);
    const { text, toolResults } = await generateText({
      model: openai('gpt-4.1-mini'),
      system: buildSystemPrompt(),
      messages,
      tools,
    });
    const toolOut = toolResultsToText(toolResults);
    const modelOut = (text && text.trim()) ? text.trim() : '';
    const out = (toolOut && toolOut.trim()) || modelOut || 'Готово!';
    messages.push({ role: 'assistant', content: out });
    await ctx.api.sendMessage(ctx.chat.id, out);
  } catch (err) {
    console.error('Handler error:', err);
    await ctx.reply('Упс, что-то пошло не так. Попробуйте ещё раз позже.');
  }
});

bot.on('message:text', async (ctx) => {
  const content = ctx.message.text.trim();
  if (!content) return;
  const chatId = ctx.chat.id;
  const messages = getHistory(chatId);
  messages.push({ role: 'user', content });
  try {
    await ctx.api.sendChatAction(ctx.chat.id, 'typing');
    const tools = buildTools(ctx.from?.id);
    const { text, toolResults } = await generateText({
      model: openai('gpt-4o-mini'),
      system: buildSystemPrompt(),
      messages,
      tools,
    });
    console.log('toolResults', toolResults);
    const toolOut = toolResultsToText(toolResults);
    const modelOut = (text && text.trim()) ? text.trim() : '';
    const out = (toolOut && toolOut.trim()) || modelOut || 'Готово!';
    messages.push({ role: 'assistant', content: out });

    await ctx.api.sendMessage(ctx.chat.id, out);
  } catch (err) {
    console.error('Handler error:', err);
    await ctx.reply('Упс, что-то пошло не так. Попробуйте ещё раз позже.');
  }
});

bot.catch(async (err) => {
  console.error('Bot error:', err);
});

bot.start().then(() => {
  console.log('Bot started.');
});
