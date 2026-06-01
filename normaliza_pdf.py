#!/usr/bin/env python3
"""Zera metadados voláteis de um PDF (CreationDate/ModDate/ID) para tornar os bytes
determinísticos entre execuções.

Motivo: o Chromium embute a data/hora de geração e um /ID aleatório no PDF → o
arquivo muda a cada run mesmo com conteúdo idêntico, gerando 1 commit/dia no git.
Com os metadados normalizados, o PDF de uma SE só muda quando o CONTEÚDO muda
(SE fecha ou correção de dado tardio) → o `git diff --cached --quiet` do cron passa
a publicar na cadência certa.

Fallback seguro: se os padrões não casarem (ex.: PDF com /Info em object stream
comprimido), o arquivo é reescrito idêntico — nada quebra, só o churn volta.
"""
import sys
import re


def normaliza(path):
    b = open(path, "rb").read()
    b = re.sub(rb"/CreationDate \(D:[^)]*\)",
               b"/CreationDate (D:20200101000000+00'00')", b)
    b = re.sub(rb"/ModDate \(D:[^)]*\)",
               b"/ModDate (D:20200101000000+00'00')", b)
    b = re.sub(rb"/ID ?\[ ?<[0-9A-Fa-f]*> ?<[0-9A-Fa-f]*> ?\]",
               b"/ID [<00000000000000000000000000000000> "
               b"<00000000000000000000000000000000>]", b)
    open(path, "wb").write(b)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("uso: python normaliza_pdf.py <arquivo.pdf>")
    normaliza(sys.argv[1])
