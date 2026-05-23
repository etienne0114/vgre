$src = 'C:\Users\Dell\Documents\vgre\scripts\vgre_sync.bat'
$text = [System.IO.File]::ReadAllText($src, [System.Text.Encoding]::UTF8)
$text = $text -replace [char]0x2014, '-'
$text = $text -replace [char]0x2013, '-'
$text = $text -replace [char]0x2500, '-'
$text = $text -replace [char]0x2192, '->'
$text = $text -replace [char]0x2019, "'"
$text = $text -replace [char]0x2018, "'"
$text = $text -replace [char]0x201C, '"'
$text = $text -replace [char]0x201D, '"'
$sb = New-Object System.Text.StringBuilder
foreach ($c in $text.ToCharArray()) {
    if ([int]$c -le 127) { [void]$sb.Append($c) } else { [void]$sb.Append('?') }
}
[System.IO.File]::WriteAllText($src, $sb.ToString(), [System.Text.Encoding]::ASCII)
$bad = ($sb.ToString().ToCharArray() | Where-Object { [int]$_ -gt 127 }).Count
Write-Host "[OK] ASCII-clean. Non-ASCII remaining: $bad"
