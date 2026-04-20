$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$htmlPath = Join-Path $workspace "Weed_Detection_Random_Forest_Report.html"
$pdfPath = Join-Path $workspace "Weed_Detection_Random_Forest_Report.pdf"

function Convert-HtmlToPlainText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Html
    )

    $bodyMatch = [regex]::Match($Html, "<body[^>]*>([\s\S]*?)</body>", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $bodyMatch.Success) {
        throw "Could not locate <body>...</body> in the HTML report."
    }

    $text = $bodyMatch.Groups[1].Value
    $text = $text -replace "(?i)<pre[^>]*>", "`n[[PRE_START]]`n"
    $text = $text -replace "(?i)</pre>", "`n[[PRE_END]]`n"
    $text = $text -replace "(?i)</t[dh]>\s*<t[dh]>", " | "
    $text = $text -replace "(?i)</tr>", "`n"
    $text = $text -replace "(?i)<li[^>]*>", "- "
    $text = $text -replace "(?i)<br\s*/?>", "`n"
    $text = $text -replace "(?i)</(p|h1|h2|h3|section|div|ul|ol|table|thead|tbody)>", "`n`n"
    $text = $text -replace "<[^>]+>", ""
    $text = [System.Net.WebUtility]::HtmlDecode($text)

    $rawLines = $text -split "`r?`n"
    $output = New-Object System.Collections.Generic.List[string]
    $inPre = $false
    $previousBlank = $true

    foreach ($rawLine in $rawLines) {
        $line = $rawLine.TrimEnd()

        if ($line -eq "[[PRE_START]]") {
            if (-not $previousBlank) {
                $output.Add("")
            }
            $inPre = $true
            $previousBlank = $false
            continue
        }

        if ($line -eq "[[PRE_END]]") {
            $inPre = $false
            $output.Add("")
            $previousBlank = $true
            continue
        }

        if ($inPre) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                if (-not $previousBlank) {
                    $output.Add("")
                    $previousBlank = $true
                }
            } else {
                $output.Add("    " + $line)
                $previousBlank = $false
            }
            continue
        }

        $line = [regex]::Replace($line, "\s+", " ").Trim()
        if ([string]::IsNullOrWhiteSpace($line)) {
            if (-not $previousBlank) {
                $output.Add("")
                $previousBlank = $true
            }
        } else {
            $output.Add($line)
            $previousBlank = $false
        }
    }

    return $output
}

function Wrap-Line {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Line,
        [int]$Width = 88
    )

    if ($Line.Length -le $Width) {
        return ,$Line
    }

    $wrapped = New-Object System.Collections.Generic.List[string]
    $isCode = $Line.StartsWith("    ")
    $indent = ""
    if ($isCode) {
        $indent = "    "
    }

    $remaining = $Line
    while ($remaining.Length -gt $Width) {
        $targetWidth = $Width
        if ($isCode) {
            $targetWidth = [Math]::Max(20, $Width - $indent.Length)
            $chunk = $remaining.Substring(0, $targetWidth)
            $wrapped.Add($chunk)
            $remaining = $indent + $remaining.Substring($targetWidth)
            continue
        }

        $breakAt = $remaining.LastIndexOf(" ", [Math]::Min($targetWidth, $remaining.Length - 1))
        if ($breakAt -lt 20) {
            $breakAt = $targetWidth
        }

        $wrapped.Add($remaining.Substring(0, $breakAt).TrimEnd())
        $remaining = $remaining.Substring($breakAt).TrimStart()
    }

    if ($remaining.Length -gt 0) {
        $wrapped.Add($remaining)
    }

    return $wrapped
}

function Escape-PdfText {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Text
    )

    $escaped = $Text.Replace("\", "\\")
    $escaped = $escaped.Replace("(", "\(")
    $escaped = $escaped.Replace(")", "\)")
    return $escaped
}

function Write-SimplePdf {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string[]]$Lines,
        [Parameter(Mandatory = $true)]
        [string]$OutputPath
    )

    $pageWidth = 595
    $pageHeight = 842
    $fontSize = 10
    $leading = 12
    $startX = 42
    $startY = 798
    $linesPerPage = 60

    $wrappedLines = New-Object System.Collections.Generic.List[string]
    foreach ($line in $Lines) {
        if ([string]::IsNullOrEmpty($line)) {
            $wrappedLines.Add("")
            continue
        }

        foreach ($segment in Wrap-Line -Line $line -Width 88) {
            $wrappedLines.Add($segment)
        }
    }

    $pageCount = [Math]::Ceiling($wrappedLines.Count / $linesPerPage)
    if ($pageCount -lt 1) {
        $pageCount = 1
    }

    $objects = New-Object System.Collections.Generic.List[string]
    $objects.Add("<< /Type /Catalog /Pages 2 0 R >>")

    $pageObjectNumbers = New-Object System.Collections.Generic.List[int]
    for ($pageIndex = 0; $pageIndex -lt $pageCount; $pageIndex++) {
        $pageObjectNumbers.Add(4 + ($pageIndex * 2))
    }

    $kids = ($pageObjectNumbers | ForEach-Object { "$_ 0 R" }) -join " "
    $objects.Add("<< /Type /Pages /Count $pageCount /Kids [ $kids ] >>")
    $objects.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    for ($pageIndex = 0; $pageIndex -lt $pageCount; $pageIndex++) {
        $pageObjectNumber = 4 + ($pageIndex * 2)
        $contentObjectNumber = 5 + ($pageIndex * 2)
        $startLine = $pageIndex * $linesPerPage
        $endLine = [Math]::Min($startLine + $linesPerPage - 1, $wrappedLines.Count - 1)
        $pageLines = if ($wrappedLines.Count -gt 0) { $wrappedLines[$startLine..$endLine] } else { @("") }

        $streamParts = New-Object System.Collections.Generic.List[string]
        $streamParts.Add("BT")
        $streamParts.Add("/F1 $fontSize Tf")
        $streamParts.Add("$startX $startY Td")
        $streamParts.Add("$leading TL")

        foreach ($pageLine in $pageLines) {
            $escapedLine = Escape-PdfText -Text $pageLine
            $streamParts.Add("($escapedLine) Tj")
            $streamParts.Add("T*")
        }

        $streamParts.Add("ET")
        $stream = ($streamParts -join "`n")
        $streamLength = [System.Text.Encoding]::ASCII.GetByteCount($stream)

        $pageObject = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 $pageWidth $pageHeight] /Resources << /Font << /F1 3 0 R >> >> /Contents $contentObjectNumber 0 R >>"
        $contentObject = "<< /Length $streamLength >>`nstream`n$stream`nendstream"

        $objects.Add($pageObject)
        $objects.Add($contentObject)
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append("%PDF-1.4`n")

    $offsets = New-Object System.Collections.Generic.List[int]
    for ($i = 0; $i -lt $objects.Count; $i++) {
        $offsets.Add([System.Text.Encoding]::ASCII.GetByteCount($builder.ToString()))
        [void]$builder.Append(($i + 1).ToString())
        [void]$builder.Append(" 0 obj`n")
        [void]$builder.Append($objects[$i])
        [void]$builder.Append("`nendobj`n")
    }

    $xrefOffset = [System.Text.Encoding]::ASCII.GetByteCount($builder.ToString())
    [void]$builder.Append("xref`n")
    [void]$builder.Append("0 ")
    [void]$builder.Append(($objects.Count + 1).ToString())
    [void]$builder.Append("`n")
    [void]$builder.Append("0000000000 65535 f `n")

    foreach ($offset in $offsets) {
        [void]$builder.AppendFormat("{0:0000000000} 00000 n `n", $offset)
    }

    [void]$builder.Append("trailer`n")
    [void]$builder.Append("<< /Size ")
    [void]$builder.Append(($objects.Count + 1).ToString())
    [void]$builder.Append(" /Root 1 0 R >>`n")
    [void]$builder.Append("startxref`n")
    [void]$builder.Append($xrefOffset.ToString())
    [void]$builder.Append("`n%%EOF")

    [System.IO.File]::WriteAllBytes($OutputPath, [System.Text.Encoding]::ASCII.GetBytes($builder.ToString()))
}

$htmlContent = Get-Content -LiteralPath $htmlPath -Raw
$plainTextLines = Convert-HtmlToPlainText -Html $htmlContent
Write-SimplePdf -Lines $plainTextLines -OutputPath $pdfPath

Get-Item -LiteralPath $pdfPath | Format-List FullName, Length, LastWriteTime
