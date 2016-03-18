dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(classifier-demo, 1.0)


AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE()

AC_PROG_CXX

ACX_PTHREAD
LIBS="$PTHREAD_LIBS $LIBS"
CXXFLAGS="$CXXFLAGS $PTHREAD_CFLAGS"

AC_CONFIG_FILES(Makefile src/Makefile tests/Makefile)
AC_OUTPUT

