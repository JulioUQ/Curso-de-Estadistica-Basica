/* Curso de Estadística Básica: cuestionarios y progreso del estudiante (F5.3).

   Se carga en todas las páginas desde assets/navegacion.html. Sin
   dependencias. Todo el contenido sigue siendo accesible sin JavaScript:
   las preguntas se ven como listas de opciones con la respuesta en un
   bloque plegable, y el progreso simplemente no aparece.

   - Cuestionarios: cada `.eb-pregunta` (data-correcta="a|b|c|d") de un
     `.eb-cuestionario` pasa a ser un grupo de botones de opción; la
     explicación del bloque «Respuesta» se muestra al comprobar.
   - Progreso: localStorage (clave eb-curso-progreso). Las lecciones son las
     páginas de los módulos y del proyecto listadas en el menú lateral; la
     clave de cada una es su ruta relativa a la raíz del libro. */
(function () {
  "use strict";

  var CLAVE = "eb-curso-progreso";
  var VERSION = 1;

  /* ---------------------------------------------------------------- */
  /* Almacenamiento tolerante: datos ausentes, corruptos o sin acceso.  */
  /* ---------------------------------------------------------------- */
  var almacen = (function () {
    try {
      var prueba = "__eb_prueba__";
      window.localStorage.setItem(prueba, "1");
      window.localStorage.removeItem(prueba);
      return window.localStorage;
    } catch (e) {
      return null;
    }
  })();

  function esObjeto(x) {
    return x !== null && typeof x === "object" && !Array.isArray(x);
  }

  function vacio() {
    return { v: VERSION, lecciones: {}, cuestionarios: {}, ultima: null };
  }

  function leer() {
    var datos = vacio();
    if (!almacen) return datos;
    try {
      var bruto = JSON.parse(almacen.getItem(CLAVE) || "null");
      if (!esObjeto(bruto)) return datos;
      // Versiones anteriores o desconocidas: se conserva lo que tenga la forma esperada.
      if (esObjeto(bruto.lecciones)) {
        Object.keys(bruto.lecciones).forEach(function (k) {
          if (bruto.lecciones[k]) datos.lecciones[k] = Number(bruto.lecciones[k].t) || Date.now();
          if (typeof bruto.lecciones[k] === "number") datos.lecciones[k] = bruto.lecciones[k];
        });
      }
      if (esObjeto(bruto.cuestionarios)) {
        Object.keys(bruto.cuestionarios).forEach(function (k) {
          var c = bruto.cuestionarios[k];
          if (esObjeto(c) && isFinite(c.aciertos) && isFinite(c.total) && c.total > 0) {
            datos.cuestionarios[k] = { aciertos: +c.aciertos, total: +c.total, t: Number(c.t) || 0 };
          }
        });
      }
      if (esObjeto(bruto.ultima) && typeof bruto.ultima.ruta === "string") {
        datos.ultima = { ruta: bruto.ultima.ruta, titulo: String(bruto.ultima.titulo || ""), t: Number(bruto.ultima.t) || 0 };
      }
    } catch (e) {
      /* datos corruptos: se empieza de cero sin romper la página */
    }
    return datos;
  }

  function guardar(datos) {
    if (!almacen) return false;
    try {
      datos.v = VERSION;
      almacen.setItem(CLAVE, JSON.stringify(datos));
      return true;
    } catch (e) {
      return false;
    }
  }

  /* ---------------------------------------------------------------- */
  /* Rutas y lecciones                                                  */
  /* ---------------------------------------------------------------- */
  var meta = document.querySelector('meta[name="quarto:offset"]');
  var raiz = new URL(meta ? meta.getAttribute("content") : "./", window.location.href);

  function rutaDe(url) {
    var u = new URL(url, window.location.href);
    var ruta = decodeURIComponent(u.pathname);
    var base = decodeURIComponent(raiz.pathname);
    if (ruta.indexOf(base) === 0) ruta = ruta.slice(base.length);
    if (ruta === "" || ruta.slice(-1) === "/") ruta += "index.html";
    return ruta;
  }

  var rutaActual = rutaDe(window.location.href);

  // Módulos y lecciones a partir del menú lateral (fuente única: _quarto.yml).
  function leerEstructura() {
    var sidebar = document.getElementById("quarto-sidebar");
    var grupos = [];
    if (!sidebar) return grupos;
    sidebar.querySelectorAll("li.sidebar-item-section").forEach(function (sec) {
      var enlaces = Array.prototype.filter.call(
        sec.querySelectorAll("a.sidebar-link[href]"),
        function (a) { return /(^|\/)notebooks\//.test(rutaDe(a.href)); }
      );
      if (!enlaces.length) return;
      var nombre = sec.querySelector(".sidebar-item-container .menu-text");
      grupos.push({
        nodo: sec,
        nombre: textoMenu(nombre),
        lecciones: enlaces.map(function (a) {
          return { ruta: rutaDe(a.href), texto: textoMenu(a.querySelector(".menu-text") || a), enlace: a };
        })
      });
    });
    return grupos;
  }

  // Texto visible de una etiqueta del menú (sin el número de capítulo oculto
  // ni las marcas que añade este script).
  function textoMenu(el) {
    if (!el) return "";
    var copia = el.cloneNode(true);
    copia.querySelectorAll(".chapter-number, .eb-menu-check, .eb-menu-count").forEach(function (n) { n.remove(); });
    return copia.textContent.replace(/\s+/g, " ").trim();
  }

  function buscarLeccion(grupos, ruta) {
    for (var i = 0; i < grupos.length; i++) {
      for (var j = 0; j < grupos[i].lecciones.length; j++) {
        if (grupos[i].lecciones[j].ruta === ruta) return { grupo: grupos[i], leccion: grupos[i].lecciones[j] };
      }
    }
    return null;
  }

  function contar(grupo, datos) {
    return grupo.lecciones.filter(function (l) { return datos.lecciones[l.ruta]; }).length;
  }

  function crear(etiqueta, clase, texto) {
    var el = document.createElement(etiqueta);
    if (clase) el.className = clase;
    if (texto != null) el.textContent = texto;
    return el;
  }

  function icono(nombre) {
    var i = crear("i", "bi bi-" + nombre);
    i.setAttribute("aria-hidden", "true");
    return i;
  }

  /* ---------------------------------------------------------------- */
  /* Menú lateral: marca de lección completada y recuento por módulo    */
  /* ---------------------------------------------------------------- */
  function pintarMenu(grupos, datos) {
    grupos.forEach(function (g) {
      g.lecciones.forEach(function (l) {
        var marca = l.enlace.querySelector(".eb-menu-check");
        if (datos.lecciones[l.ruta]) {
          if (!marca) {
            marca = crear("span", "eb-menu-check");
            marca.appendChild(icono("check-circle-fill"));
            marca.appendChild(crear("span", "visually-hidden", " (completada)"));
            l.enlace.appendChild(marca);
          }
        } else if (marca) {
          marca.remove();
        }
      });
      var cabecera = g.nodo.querySelector(":scope > .sidebar-item-container .sidebar-item-text") ||
                     g.nodo.querySelector(":scope > .sidebar-item-container");
      if (!cabecera) return;
      var cuenta = cabecera.querySelector(".eb-menu-count");
      var hechas = contar(g, datos);
      if (!hechas) {
        if (cuenta) cuenta.remove();
        return;
      }
      if (!cuenta) {
        // Dentro de la etiqueta: si no cabe en la línea, baja a la siguiente
        // en lugar de recortar el nombre del módulo.
        cuenta = crear("span", "eb-menu-count");
        (cabecera.querySelector(".menu-text") || cabecera).appendChild(cuenta);
      }
      cuenta.textContent = hechas + "/" + g.lecciones.length;
      cuenta.setAttribute("aria-label", hechas + " de " + g.lecciones.length + " lecciones completadas");
      cuenta.classList.toggle("eb-menu-count-done", hechas === g.lecciones.length);
    });
  }

  /* ---------------------------------------------------------------- */
  /* Tarjeta «Marcar como completada» al final de cada lección           */
  /* ---------------------------------------------------------------- */
  function tarjetaLeccion(grupos, actual, datos) {
    var main = document.querySelector("main#quarto-document-content") || document.querySelector("main.content");
    if (!main) return;
    var caja = crear("section", "eb-lesson-status");
    caja.setAttribute("aria-label", "Progreso de esta lección");
    var texto = crear("div", "eb-lesson-status-text");
    var titulo = crear("p", "eb-lesson-status-title");
    var detalle = crear("p", "eb-lesson-status-detail");
    texto.appendChild(titulo);
    texto.appendChild(detalle);
    var boton = crear("button", "eb-btn");
    boton.type = "button";
    caja.appendChild(texto);
    caja.appendChild(boton);
    main.appendChild(caja);

    function pintar() {
      var hecha = !!datos.lecciones[actual.leccion.ruta];
      var n = contar(actual.grupo, datos);
      caja.classList.toggle("is-done", hecha);
      titulo.textContent = hecha ? "Lección completada" : "¿Has terminado esta lección?";
      titulo.insertBefore(icono(hecha ? "check-circle-fill" : "circle"), titulo.firstChild);
      detalle.textContent = actual.grupo.nombre + ": " + n + " de " + actual.grupo.lecciones.length +
        (actual.grupo.lecciones.length === 1 ? " lección completada." : " lecciones completadas.");
      boton.textContent = hecha ? "Desmarcar" : "Marcar como completada";
      boton.className = "eb-btn " + (hecha ? "eb-btn-secondary" : "eb-btn-primary");
      boton.setAttribute("aria-pressed", hecha ? "true" : "false");
      if (!almacen) {
        boton.hidden = true;
        titulo.textContent = "Progreso no disponible";
        detalle.textContent = "Este navegador no permite guardar datos de la página (almacenamiento local desactivado " +
          "o bloqueado), así que no se puede registrar el progreso. El contenido y los cuestionarios funcionan igual.";
      }
    }

    boton.addEventListener("click", function () {
      datos = leer();
      if (datos.lecciones[actual.leccion.ruta]) delete datos.lecciones[actual.leccion.ruta];
      else datos.lecciones[actual.leccion.ruta] = Date.now();
      guardar(datos);
      pintar();
      pintarMenu(grupos, datos);
    });
    pintar();
  }

  /* ---------------------------------------------------------------- */
  /* Cuestionarios                                                      */
  /* ---------------------------------------------------------------- */
  var contadorCuestionarios = 0;

  function prepararCuestionario(cont, datos) {
    var preguntas = Array.prototype.slice.call(cont.querySelectorAll(":scope > .eb-pregunta"));
    if (!preguntas.length) return;
    var id = "ebq" + (++contadorCuestionarios);
    var modelo = [];

    var ayuda = cont.querySelector(":scope > .eb-cuestionario-ayuda");
    if (ayuda) {
      ayuda.textContent = "Elige una opción en cada pregunta y pulsa «Comprobar respuestas». " +
        "Verás una explicación de cada una y podrás repetir el cuestionario.";
    }

    var valido = preguntas.every(function (p, i) {
      var lista = p.querySelector(":scope > ol");
      var correcta = (p.getAttribute("data-correcta") || "").trim().toLowerCase();
      var enunciado = lista && lista.previousElementSibling;
      var callout = p.querySelector(":scope > .callout");
      var cuerpo = callout && callout.querySelector(".callout-body-container");
      var opciones = lista ? lista.querySelectorAll(":scope > li") : [];
      var indice = "abcd".indexOf(correcta);
      if (!lista || !enunciado || !cuerpo || indice < 0 || indice >= opciones.length) return false;
      modelo.push({ nodo: p, lista: lista, enunciado: enunciado, callout: callout, cuerpo: cuerpo,
                    opciones: opciones, correcta: indice, n: i });
      return true;
    });
    if (!valido) return; // marcado inesperado: se deja la versión sin JavaScript

    modelo.forEach(function (q) {
      var campo = crear("fieldset", "eb-q");
      var leyenda = crear("legend", "eb-q-stem");
      while (q.enunciado.firstChild) leyenda.appendChild(q.enunciado.firstChild);
      campo.appendChild(leyenda);

      var grupoOpciones = crear("div", "eb-q-options");
      q.radios = [];
      q.etiquetas = [];
      Array.prototype.forEach.call(q.opciones, function (li, k) {
        var etiqueta = crear("label", "eb-q-option");
        var radio = crear("input");
        radio.type = "radio";
        radio.name = id + "-" + q.n;
        radio.value = String(k);
        var letra = crear("span", "eb-q-letter", "abcd".charAt(k));
        letra.setAttribute("aria-hidden", "true");
        var texto = crear("span", "eb-q-text");
        var origen = li.children.length === 1 && li.firstElementChild.tagName === "P" ? li.firstElementChild : li;
        while (origen.firstChild) texto.appendChild(origen.firstChild);
        var estado = crear("span", "eb-q-mark");
        etiqueta.appendChild(radio);
        etiqueta.appendChild(letra);
        etiqueta.appendChild(texto);
        etiqueta.appendChild(estado);
        grupoOpciones.appendChild(etiqueta);
        q.radios.push(radio);
        q.etiquetas.push(etiqueta);
      });
      campo.appendChild(grupoOpciones);

      // La explicación existente pasa al bloque de respuesta (sin duplicarla).
      var respuesta = crear("div", "eb-q-feedback");
      respuesta.hidden = true;
      q.veredicto = crear("p", "eb-q-verdict");
      respuesta.appendChild(q.veredicto);
      var clave = q.cuerpo.querySelector(".eb-clave");
      if (clave) clave.remove();
      while (q.cuerpo.firstChild) respuesta.appendChild(q.cuerpo.firstChild);
      var primero = respuesta.querySelector(".eb-q-verdict + p");
      if (primero && primero.firstChild && primero.firstChild.nodeType === 3) {
        primero.firstChild.nodeValue = primero.firstChild.nodeValue.replace(/^\s+/, "");
      }
      campo.appendChild(respuesta);
      q.respuesta = respuesta;
      q.campo = campo;

      q.lista.remove();
      q.callout.remove();
      if (!q.enunciado.childNodes.length) q.enunciado.remove();
      q.nodo.appendChild(campo);
    });

    // Acciones y resumen
    var pie = crear("div", "eb-quiz-actions");
    var comprobar = crear("button", "eb-btn eb-btn-primary", "Comprobar respuestas");
    comprobar.type = "button";
    var repetir = crear("button", "eb-btn eb-btn-secondary", "Repetir cuestionario");
    repetir.type = "button";
    repetir.hidden = true;
    var resumen = crear("p", "eb-quiz-summary");
    resumen.setAttribute("role", "status");
    resumen.setAttribute("aria-live", "polite");
    var anterior = crear("p", "eb-quiz-last");
    pie.appendChild(comprobar);
    pie.appendChild(repetir);
    cont.appendChild(pie);
    cont.appendChild(resumen);
    cont.appendChild(anterior);

    function pintarAnterior() {
      var r = datos.cuestionarios[rutaActual];
      anterior.textContent = r ? "Último intento guardado: " + r.aciertos + " de " + r.total + " respuestas correctas." : "";
      anterior.hidden = !r;
    }
    pintarAnterior();

    comprobar.addEventListener("click", function () {
      var sinResponder = modelo.filter(function (q) {
        return !q.radios.some(function (r) { return r.checked; });
      });
      if (sinResponder.length) {
        resumen.className = "eb-quiz-summary is-warning";
        resumen.textContent = "";
        resumen.appendChild(icono("exclamation-circle"));
        resumen.appendChild(document.createTextNode(" Te falta" + (sinResponder.length > 1 ? "n " : " ") +
          sinResponder.length + " pregunta" + (sinResponder.length > 1 ? "s" : "") + " por responder."));
        sinResponder.forEach(function (q) { q.campo.classList.add("is-missing"); });
        sinResponder[0].radios[0].focus();
        return;
      }
      var aciertos = 0;
      modelo.forEach(function (q) {
        var elegida = q.radios.findIndex(function (r) { return r.checked; });
        var bien = elegida === q.correcta;
        if (bien) aciertos++;
        q.campo.classList.remove("is-missing");
        q.campo.classList.add(bien ? "is-correct" : "is-incorrect");
        q.radios.forEach(function (r) { r.disabled = true; });
        q.etiquetas.forEach(function (et, k) {
          var marca = et.querySelector(".eb-q-mark");
          marca.textContent = "";
          if (k === q.correcta) {
            et.classList.add("is-answer");
            marca.appendChild(icono("check-circle-fill"));
            marca.appendChild(document.createTextNode(k === elegida ? " Tu respuesta" : " Respuesta correcta"));
          } else if (k === elegida) {
            et.classList.add("is-wrong");
            marca.appendChild(icono("x-circle-fill"));
            marca.appendChild(document.createTextNode(" Tu respuesta"));
          }
        });
        q.veredicto.textContent = "";
        q.veredicto.appendChild(icono(bien ? "check-circle-fill" : "x-circle-fill"));
        q.veredicto.appendChild(document.createTextNode(bien
          ? " Correcto."
          : " Incorrecto. La respuesta correcta es la " + "abcd".charAt(q.correcta) + "."));
        q.respuesta.hidden = false;
      });
      var total = modelo.length;
      resumen.className = "eb-quiz-summary is-result";
      resumen.textContent = "";
      resumen.appendChild(icono(aciertos === total ? "trophy" : "clipboard-check"));
      resumen.appendChild(document.createTextNode(" " + aciertos + " de " + total + " respuestas correctas." +
        (aciertos === total ? " ¡Muy bien!" : " Revisa las explicaciones y vuelve a intentarlo cuando quieras.")));
      comprobar.hidden = true;
      repetir.hidden = false;
      datos = leer();
      datos.cuestionarios[rutaActual] = { aciertos: aciertos, total: total, t: Date.now() };
      guardar(datos);
      pintarAnterior();
      repetir.focus();
    });

    repetir.addEventListener("click", function () {
      modelo.forEach(function (q) {
        q.campo.classList.remove("is-correct", "is-incorrect", "is-missing");
        q.radios.forEach(function (r) { r.disabled = false; r.checked = false; });
        q.etiquetas.forEach(function (et) {
          et.classList.remove("is-answer", "is-wrong");
          et.querySelector(".eb-q-mark").textContent = "";
        });
        q.respuesta.hidden = true;
      });
      resumen.textContent = "";
      resumen.className = "eb-quiz-summary";
      repetir.hidden = true;
      comprobar.hidden = false;
      modelo[0].radios[0].focus();
    });

    cont.classList.add("is-interactive");
  }

  /* ---------------------------------------------------------------- */
  /* Portada: «Tu progreso»                                             */
  /* ---------------------------------------------------------------- */
  function panelPortada(grupos, datos) {
    var panel = document.getElementById("eb-progreso");
    if (!panel || !almacen) return;
    var total = 0, hechas = 0;
    grupos.forEach(function (g) { total += g.lecciones.length; hechas += contar(g, datos); });
    var encontrada = datos.ultima && buscarLeccion(grupos, datos.ultima.ruta);
    var ultima = encontrada
      ? { ruta: encontrada.leccion.ruta, titulo: encontrada.grupo.nombre + " · " + encontrada.leccion.texto }
      : null;
    if (!total || (!hechas && !ultima)) { panel.hidden = true; return; }

    var pct = Math.round((100 * hechas) / total);
    panel.textContent = "";
    var cab = crear("div", "eb-progress-head");
    var titulo = crear("h2", "eb-progress-title", "Tu progreso");
    titulo.id = "eb-progreso-titulo";
    cab.appendChild(titulo);
    cab.appendChild(crear("p", "eb-progress-figure", hechas + " de " + total + " lecciones completadas (" + pct + " %)"));
    panel.appendChild(cab);

    var barra = crear("div", "eb-progress-bar");
    barra.setAttribute("role", "progressbar");
    barra.setAttribute("aria-valuemin", "0");
    barra.setAttribute("aria-valuemax", "100");
    barra.setAttribute("aria-valuenow", String(pct));
    barra.setAttribute("aria-labelledby", "eb-progreso-titulo");
    var relleno = crear("span");
    relleno.style.width = pct + "%";
    barra.appendChild(relleno);
    panel.appendChild(barra);

    var lista = crear("ul", "eb-progress-modules");
    grupos.forEach(function (g) {
      var n = contar(g, datos);
      var li = crear("li", n === g.lecciones.length ? "is-done" : (n ? "is-started" : ""));
      li.appendChild(crear("span", "eb-progress-module-name", g.nombre));
      var cifra = crear("span", "eb-progress-module-count", n + "/" + g.lecciones.length);
      cifra.setAttribute("aria-label", n + " de " + g.lecciones.length + " lecciones");
      li.appendChild(cifra);
      var teoria = g.lecciones.filter(function (l) { return datos.cuestionarios[l.ruta]; })[0];
      if (teoria) {
        var r = datos.cuestionarios[teoria.ruta];
        li.appendChild(crear("span", "eb-progress-module-quiz", "Cuestionario: " + r.aciertos + "/" + r.total));
      }
      lista.appendChild(li);
    });
    panel.appendChild(lista);

    var acciones = crear("div", "eb-progress-actions");
    if (ultima) {
      var seguir = crear("a", "eb-btn eb-btn-primary");
      seguir.href = new URL(ultima.ruta, raiz).href;
      seguir.appendChild(document.createTextNode("Continuar: " + (ultima.titulo || "última lección") + " "));
      seguir.appendChild(icono("arrow-right"));
      acciones.appendChild(seguir);
    }
    var reiniciar = crear("button", "eb-btn eb-btn-secondary", "Reiniciar progreso");
    reiniciar.type = "button";
    reiniciar.addEventListener("click", function () {
      if (!window.confirm("¿Seguro que quieres borrar tu progreso (lecciones completadas, resultados de los cuestionarios y última página visitada) en este navegador? No se puede deshacer.")) return;
      try { almacen.removeItem(CLAVE); } catch (e) { /* nada que borrar */ }
      datos = leer();
      pintarMenu(grupos, datos);
      panel.hidden = true;
      var destino = document.getElementById("modulos");
      if (destino) { destino.setAttribute("tabindex", "-1"); destino.focus(); }
    });
    acciones.appendChild(reiniciar);
    panel.appendChild(acciones);
    panel.appendChild(crear("p", "eb-progress-note",
      "Tu progreso se guarda solo en este navegador: no se envía a ningún servidor ni se comparte entre dispositivos."));
    panel.hidden = false;
  }

  /* ---------------------------------------------------------------- */
  function iniciar() {
    var datos = leer();
    var grupos = leerEstructura();
    var actual = buscarLeccion(grupos, rutaActual);

    if (actual && almacen) {
      datos.ultima = { ruta: actual.leccion.ruta, titulo: actual.grupo.nombre + " · " + actual.leccion.texto, t: Date.now() };
      guardar(datos);
    }
    document.querySelectorAll(".eb-cuestionario").forEach(function (c) { prepararCuestionario(c, datos); });
    if (actual) tarjetaLeccion(grupos, actual, datos);
    pintarMenu(grupos, datos);
    panelPortada(grupos, datos);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", iniciar);
  else iniciar();
})();
