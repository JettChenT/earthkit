/**
 * @param {string} value
 * @returns {RegExp}
 * */

/**
 * @param {RegExp | string } re
 * @returns {string}
 */
function source(re) {
  if (!re) return null;
  if (typeof re === "string") return re;

  return re.source;
}

/**
 * @param {RegExp | string } re
 * @returns {string}
 */
function lookahead(re) {
  return concat("(?=", re, ")");
}

/**
 * @param {RegExp | string } re
 * @returns {string}
 */
function optional(re) {
  return concat("(", re, ")?");
}

/**
 * @param {...(RegExp | string) } args
 * @returns {string}
 */
function concat(...args) {
  const joined = args.map((x) => source(x)).join("");
  return joined;
}

/*
  Language: Overpass Turbo
  Category: common, system
  Website: https://overpass-turbo.eu
  */

/** @type LanguageFn */
function overpassTurbo(hljs) {
  const COMMENT_MODE = hljs.COMMENT("//", "$", {
    contains: [
      {
        begin: /\\\n/,
      },
    ],
  });
  const KEYWORDS = {
    keyword: "out body meta qt asc bbox around user uid newer changed",
    literal: "true false",
    built_in: "node way relation area",
  };

  const STRINGS = {
    className: "string",
    variants: [
      {
        begin: '"',
        end: '"',
        illegal: "\\n",
        contains: [hljs.BACKSLASH_ESCAPE],
      },
      {
        begin: "'",
        end: "'",
        illegal: "\\n",
        contains: [hljs.BACKSLASH_ESCAPE],
      },
    ],
  };

  const NUMBERS = {
    className: "number",
    variants: [
      {
        begin: "\\b\\d+(\\.\\d+)?",
      },
    ],
    relevance: 0,
  };

  const FUNCTION_TITLE = hljs.IDENT_RE + "\\s*\\(";

  const FUNCTION_DECLARATION = {
    className: "function",
    begin: FUNCTION_TITLE,
    returnBegin: true,
    end: /[{;=]/,
    excludeEnd: true,
    keywords: KEYWORDS,
    illegal: /[^\w\s\*&:<>.]/,
    contains: [
      {
        begin: hljs.IDENT_RE,
        className: "title",
        relevance: 0,
      },
      {
        className: "params",
        begin: /\(/,
        end: /\)/,
        keywords: KEYWORDS,
        relevance: 0,
        contains: [COMMENT_MODE, STRINGS, NUMBERS],
      },
      COMMENT_MODE,
      STRINGS,
      NUMBERS,
    ],
  };

  return {
    name: "Overpass Turbo",
    aliases: ["overpass", "osm"],
    keywords: KEYWORDS,
    illegal: "</",
    contains: [COMMENT_MODE, FUNCTION_DECLARATION, STRINGS, NUMBERS],
  };
}

export default overpassTurbo;
