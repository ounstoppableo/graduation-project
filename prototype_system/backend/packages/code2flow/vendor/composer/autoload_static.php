<?php

// autoload_static.php @generated by Composer

namespace Composer\Autoload;

class ComposerStaticInita3140b8181412c210962c964a62db586
{
    public static $prefixLengthsPsr4 = array (
        'P' => 
        array (
            'PhpParser\\' => 10,
        ),
    );

    public static $prefixDirsPsr4 = array (
        'PhpParser\\' => 
        array (
            0 => __DIR__ . '/..' . '/nikic/php-parser/lib/PhpParser',
        ),
    );

    public static $classMap = array (
        'Composer\\InstalledVersions' => __DIR__ . '/..' . '/composer/InstalledVersions.php',
    );

    public static function getInitializer(ClassLoader $loader)
    {
        return \Closure::bind(function () use ($loader) {
            $loader->prefixLengthsPsr4 = ComposerStaticInita3140b8181412c210962c964a62db586::$prefixLengthsPsr4;
            $loader->prefixDirsPsr4 = ComposerStaticInita3140b8181412c210962c964a62db586::$prefixDirsPsr4;
            $loader->classMap = ComposerStaticInita3140b8181412c210962c964a62db586::$classMap;

        }, null, ClassLoader::class);
    }
}