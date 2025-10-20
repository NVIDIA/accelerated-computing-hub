#pragma once
//#define DISABLE_CART_PROD_IOTA_SPEC
/*
Adapted from TartanLlama/ranges: https://github.com/TartanLlama/ranges
Original version License CC0 1.0 Universal (see below)

Modified by Gonzalo Brito Gadeschi, NVIDIA corporation
Modifications under MIT license.

---

SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

---

Creative Commons Legal Code

CC0 1.0 Universal

    CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM AND DOES NOT PROVIDE
    LEGAL SERVICES. DISTRIBUTION OF THIS DOCUMENT DOES NOT CREATE AN
    ATTORNEY-CLIENT RELATIONSHIP. CREATIVE COMMONS PROVIDES THIS
    INFORMATION ON AN "AS-IS" BASIS. CREATIVE COMMONS MAKES NO WARRANTIES
    REGARDING THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS
    PROVIDED HEREUNDER, AND DISCLAIMS LIABILITY FOR DAMAGES RESULTING FROM
    THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS PROVIDED
    HEREUNDER.

Statement of Purpose

The laws of most jurisdictions throughout the world automatically confer
exclusive Copyright and Related Rights (defined below) upon the creator
and subsequent owner(s) (each and all, an "owner") of an original work of
authorship and/or a database (each, a "Work").

Certain owners wish to permanently relinquish those rights to a Work for
the purpose of contributing to a commons of creative, cultural and
scientific works ("Commons") that the public can reliably and without fear
of later claims of infringement build upon, modify, incorporate in other
works, reuse and redistribute as freely as possible in any form whatsoever
and for any purposes, including without limitation commercial purposes.
These owners may contribute to the Commons to promote the ideal of a free
culture and the further production of creative, cultural and scientific
works, or to gain reputation or greater distribution for their Work in
part through the use and efforts of others.

For these and/or other purposes and motivations, and without any
expectation of additional consideration or compensation, the person
associating CC0 with a Work (the "Affirmer"), to the extent that he or she
is an owner of Copyright and Related Rights in the Work, voluntarily
elects to apply CC0 to the Work and publicly distribute the Work under its
terms, with knowledge of his or her Copyright and Related Rights in the
Work and the meaning and intended legal effect of CC0 on those rights.

1. Copyright and Related Rights. A Work made available under CC0 may be
protected by copyright and related or neighboring rights ("Copyright and
Related Rights"). Copyright and Related Rights include, but are not
limited to, the following:

  i. the right to reproduce, adapt, distribute, perform, display,
     communicate, and translate a Work;
 ii. moral rights retained by the original author(s) and/or performer(s);
iii. publicity and privacy rights pertaining to a person's image or
     likeness depicted in a Work;
 iv. rights protecting against unfair competition in regards to a Work,
     subject to the limitations in paragraph 4(a), below;
  v. rights protecting the extraction, dissemination, use and reuse of data
     in a Work;
 vi. database rights (such as those arising under Directive 96/9/EC of the
     European Parliament and of the Council of 11 March 1996 on the legal
     protection of databases, and under any national implementation
     thereof, including any amended or successor version of such
     directive); and
vii. other similar, equivalent or corresponding rights throughout the
     world based on applicable law or treaty, and any national
     implementations thereof.

2. Waiver. To the greatest extent permitted by, but not in contravention
of, applicable law, Affirmer hereby overtly, fully, permanently,
irrevocably and unconditionally waives, abandons, and surrenders all of
Affirmer's Copyright and Related Rights and associated claims and causes
of action, whether now known or unknown (including existing as well as
future claims and causes of action), in the Work (i) in all territories
worldwide, (ii) for the maximum duration provided by applicable law or
treaty (including future time extensions), (iii) in any current or future
medium and for any number of copies, and (iv) for any purpose whatsoever,
including without limitation commercial, advertising or promotional
purposes (the "Waiver"). Affirmer makes the Waiver for the benefit of each
member of the public at large and to the detriment of Affirmer's heirs and
successors, fully intending that such Waiver shall not be subject to
revocation, rescission, cancellation, termination, or any other legal or
equitable action to disrupt the quiet enjoyment of the Work by the public
as contemplated by Affirmer's express Statement of Purpose.

3. Public License Fallback. Should any part of the Waiver for any reason
be judged legally invalid or ineffective under applicable law, then the
Waiver shall be preserved to the maximum extent permitted taking into
account Affirmer's express Statement of Purpose. In addition, to the
extent the Waiver is so judged Affirmer hereby grants to each affected
person a royalty-free, non transferable, non sublicensable, non exclusive,
irrevocable and unconditional license to exercise Affirmer's Copyright and
Related Rights in the Work (i) in all territories worldwide, (ii) for the
maximum duration provided by applicable law or treaty (including future
time extensions), (iii) in any current or future medium and for any number
of copies, and (iv) for any purpose whatsoever, including without
limitation commercial, advertising or promotional purposes (the
"License"). The License shall be deemed effective as of the date CC0 was
applied by Affirmer to the Work. Should any part of the License for any
reason be judged legally invalid or ineffective under applicable law, such
partial invalidity or ineffectiveness shall not invalidate the remainder
of the License, and in such case Affirmer hereby affirms that he or she
will not (i) exercise any of his or her remaining Copyright and Related
Rights in the Work or (ii) assert any associated claims and causes of
action with respect to the Work, in either case contrary to Affirmer's
express Statement of Purpose.

4. Limitations and Disclaimers.

 a. No trademark or patent rights held by Affirmer are waived, abandoned,
    surrendered, licensed or otherwise affected by this document.
 b. Affirmer offers the Work as-is and makes no representations or
    warranties of any kind concerning the Work, express, implied,
    statutory or otherwise, including without limitation warranties of
    title, merchantability, fitness for a particular purpose, non
    infringement, or the absence of latent or other defects, accuracy, or
    the present or absence of errors, whether or not discoverable, all to
    the greatest extent permissible under applicable law.
 c. Affirmer disclaims responsibility for clearing rights of other persons
    that may apply to the Work or any use thereof, including without
    limitation any person's Copyright and Related Rights in the Work.
    Further, Affirmer disclaims responsibility for obtaining any necessary
    consents, permissions or other rights required for any use of the
    Work.
 d. Affirmer understands and acknowledges that Creative Commons is not a
    party to this document and has no duty or obligation with respect to
    this CC0 or use of the Work.
*/

#include <algorithm>
#include <concepts>
#include <functional>
#include <iterator>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tl {
  namespace detail {
    template <class I>
    concept single_pass_iterator = std::input_or_output_iterator<I> && !std::forward_iterator<I>;

    template <typename... V>
    constexpr auto common_iterator_category() {
      if constexpr ((std::ranges::random_access_range<V> && ...))
        return std::random_access_iterator_tag{};
      else if constexpr ((std::ranges::bidirectional_range<V> && ...))
        return std::bidirectional_iterator_tag{};
      else if constexpr ((std::ranges::forward_range<V> && ...))
        return std::forward_iterator_tag{};
      else if constexpr ((std::ranges::input_range<V> && ...))
        return std::input_iterator_tag{};
      else
        return std::output_iterator_tag{};
    }
  }

  template <class... V>
  using common_iterator_category = decltype(detail::common_iterator_category<V...>());

  template <class R>
  concept simple_view = std::ranges::view<R> && std::ranges::range<const R> &&
    std::same_as<std::ranges::iterator_t<R>, std::ranges::iterator_t<const R>> &&
    std::same_as<std::ranges::sentinel_t<R>,
    std::ranges::sentinel_t<const R>>;

  struct as_sentinel_t {};
  constexpr inline as_sentinel_t as_sentinel;

  template <bool Const, class T>
  using maybe_const = std::conditional_t<Const, const T, T>;

  template <std::destructible T>
  class basic_mixin : protected T {
  public:
    constexpr basic_mixin()
      noexcept(std::is_nothrow_default_constructible<T>::value)
      requires std::default_initializable<T> :
      T() {}
    constexpr basic_mixin(const T& t)
      noexcept(std::is_nothrow_copy_constructible<T>::value)
      requires std::copy_constructible<T> :
      T(t) {}
    constexpr basic_mixin(T&& t)
      noexcept(std::is_nothrow_move_constructible<T>::value)
      requires std::move_constructible<T> :
      T(std::move(t)) {}


    constexpr T& get() & noexcept { return *static_cast<T*>(this); }
    constexpr const T& get() const& noexcept { return *static_cast<T const*>(this); }
    constexpr T&& get() && noexcept { return std::move(*static_cast<T*>(this)); }
    constexpr const T&& get() const&& noexcept { return std::move(*static_cast<T const*>(this)); }
  };

  namespace cursor {
    namespace detail {
      template <class C>
      struct tags {
        static constexpr auto single_pass() requires requires { { C::single_pass } -> std::convertible_to<bool>; } {
          return C::single_pass;
        }
        static constexpr auto single_pass() { return false; }

        static constexpr auto contiguous() requires requires { { C::contiguous } -> std::convertible_to<bool>; } {
          return C::contiguous;
        }
        static constexpr auto contiguous() { return false; }
      };
    }
    template <class C>
    constexpr bool single_pass = detail::tags<C>::single_pass();

    template <class C>
    constexpr bool tagged_contiguous = detail::tags<C>::contiguous();

    namespace detail {
      template <class C>
      struct deduced_mixin_t {
        template <class T> static auto deduce(int)-> typename T::mixin;
        template <class T> static auto deduce(...)->tl::basic_mixin<T>;
        using type = decltype(deduce<C>(0));
      };
    }

    template <class C>
    using mixin_t = typename detail::deduced_mixin_t<C>::type;

    template <class C>
    requires
      requires(const C& c) { c.read(); }
    using reference_t = decltype(std::declval<const C&>().read());

    namespace detail {
      template <class C>
      struct deduced_value_t {
        template<class T> static auto deduce(int)-> typename T::value_type;
        template<class T> static auto deduce(...)->std::decay_t<reference_t<T>>;

        using type = decltype(deduce<C>(0));
      };
    }

    template <class C>
    requires std::same_as<typename detail::deduced_value_t<C>::type, std::decay_t<typename detail::deduced_value_t<C>::type>>
      using value_type_t = typename detail::deduced_value_t<C>::type;

    namespace detail {
      template <class C>
      struct deduced_difference_t {
        template <class T> static auto deduce(int)-> typename T::difference_type;
        template <class T>
        static auto deduce(long)->decltype(std::declval<const T&>().distance_to(std::declval<const T&>()));
        template <class T>
        static auto deduce(...)->std::ptrdiff_t;

        using type = decltype(deduce<C>(0));
      };
    }

    template <class C>
    using difference_type_t = typename detail::deduced_difference_t<C>::type;

    template <class C>
    concept cursor = std::semiregular<std::remove_cv_t<C>>
      && std::semiregular<mixin_t<std::remove_cv_t<C>>>
      && requires {typename difference_type_t<C>; };

    template <class C>
    concept readable = cursor<C> && requires(const C & c) {
      c.read();
      typename reference_t<C>;
      typename value_type_t<C>;
    };

    template <class C>
    concept arrow = readable<C>
      && requires(const C & c) { c.arrow(); };

    template <class C, class T>
    concept writable = cursor<C>
      && requires(C & c, T && t) { c.write(std::forward<T>(t)); };

    template <class S, class C>
    concept sentinel_for = cursor<C> && std::semiregular<S>
      && requires(const C & c, const S & s) { {c.equal(s)} -> std::same_as<bool>; };

    template <class S, class C>
    concept sized_sentinel_for = sentinel_for<S, C> &&
      requires(const C & c, const S & s) {
        {c.distance_to(s)} -> std::same_as<difference_type_t<C>>;
    };

    template <class C>
    concept next = cursor<C> && requires(C & c) { c.next(); };

    template <class C>
    concept prev = cursor<C> && requires(C & c) { c.prev(); };

    template <class C>
    concept advance = cursor<C>
      && requires(C & c, difference_type_t<C> n) { c.advance(n); };

    template <class C>
    concept indirect_move = readable<C>
      && requires(const C & c) { c.indirect_move(); };

    template <class C, class O>
    concept indirect_swap = readable<C> && readable<O>
      && requires(const C & c, const O & o) {
      c.indirect_swap(o);
      o.indirect_swap(c);
    };

    template <class C>
    concept input = readable<C> && next<C>;
    template <class C>
    concept forward = input<C> && sentinel_for<C, C> && !single_pass<C>;
    template <class C>
    concept bidirectional = forward<C> && prev<C>;
    template <class C>
    concept random_access = bidirectional<C> && advance<C> && sized_sentinel_for<C, C>;
    template <class C>
    concept contiguous = random_access<C> && tagged_contiguous<C> && std::is_reference_v<reference_t<C>>;

    template <class C>
    constexpr auto cpp20_iterator_category() {
      if constexpr (contiguous<C>)
        return std::contiguous_iterator_tag{};
      else if constexpr (random_access<C>)
        return std::random_access_iterator_tag{};
      else if constexpr (bidirectional<C>)
        return std::bidirectional_iterator_tag{};
      else if constexpr (forward<C>)
        return std::forward_iterator_tag{};
      else
        return std::input_iterator_tag{};
    }
    template <class C>
    using cpp20_iterator_category_t = decltype(cpp20_iterator_category<C>());

    //There were a few changes in requirements on iterators between C++17 and C++20
    //See https://wg21.link/p2259 for discussion
    //- C++17 didn't have contiguous iterators
    //- C++17 input iterators required *it++ to be valid
    //- C++17 forward iterators required the reference type to be exactly value_type&/value_type const& (i.e. not a proxy)
    struct not_a_cpp17_iterator {};

    template <class C>
    concept reference_is_value_type_ref =
      (std::same_as<reference_t<C>, value_type_t<C>&> || std::same_as<reference_t<C>, value_type_t<C> const&>);

    template <class C>
    concept can_create_postincrement_proxy =
      (std::move_constructible<value_type_t<C>> && std::constructible_from<value_type_t<C>, reference_t<C>>);

    template <class C>
    constexpr auto cpp17_iterator_category() {
      if constexpr (random_access<C>
#if !defined(__NVCOMPILER)
                    // YOLO: with nvc++ proxy iterators can be random access . . .
                    // BUG: Need to update Thrust to C++20 iterator categories
                    && reference_is_value_type_ref<C>
#endif
      )
        return std::random_access_iterator_tag{};
      else if constexpr (bidirectional<C> && reference_is_value_type_ref<C>)
        return std::bidirectional_iterator_tag{};
      else if constexpr (forward<C> && reference_is_value_type_ref<C>)
        return std::forward_iterator_tag{};
      else if constexpr (can_create_postincrement_proxy<C>)
        return std::input_iterator_tag{};
      else
        return not_a_cpp17_iterator{};
    }
    template <class C>
    using cpp17_iterator_category_t = decltype(cpp17_iterator_category<C>());

    //iterator_concept and iterator_category are tricky; this abstracts them out.
    //Since the rules for iterator categories changed between C++17 and C++20
    //a C++20 iterator may have a weaker category in C++17,
    //or it might not be a valid C++17 iterator at all.
    //iterator_concept will be the C++20 iterator category.
    //iterator_category will be the C++17 iterator category, or it will not exist
    //in the case that the iterator is not a valid C++17 iterator.
    template <cursor C, class category = cpp17_iterator_category_t<C>>
    struct associated_types_category_base {
      using iterator_category = category;
    };
    template <cursor C>
    struct associated_types_category_base<C, not_a_cpp17_iterator> {};

    template <cursor C>
    struct associated_types : associated_types_category_base<C> {
      using iterator_concept = cpp20_iterator_category_t<C>;
      using value_type = cursor::value_type_t<C>;
      using difference_type = cursor::difference_type_t<C>;
      using reference = cursor::reference_t<C>;
    };

    namespace detail {
      // We assume a cursor is writeable if it's either not readable
      // or it is writeable with the same type it reads to
      template <class C>
      struct is_writable_cursor {
        template <readable T>
        requires requires (C c) {
          c.write(c.read());
        }
        static auto deduce()->std::true_type;

        template <readable T>
        static auto deduce()->std::false_type;

        template <class T>
        static auto deduce()->std::true_type;

        static constexpr bool value = decltype(deduce<C>())::value;
      };
    }
  }

  namespace detail {
    template <class T>
    struct post_increment_proxy {
    private:
      T cache_;

    public:
      template<typename U>
      constexpr post_increment_proxy(U&& t)
        : cache_(std::forward<U>(t))
      {}
      constexpr T const& operator*() const noexcept
      {
        return cache_;
      }
    };
  }


  template <cursor::input C>
  class basic_iterator :
    public cursor::mixin_t<C>
  {
  private:
    using mixin = cursor::mixin_t<C>;

    constexpr auto& cursor() noexcept { return this->mixin::get(); }
    constexpr auto const& cursor() const noexcept { return this->mixin::get(); }

    template <cursor::input>
    friend class basic_iterator;

    //TODO these need to change to support output iterators
    using reference_t = decltype(std::declval<C>().read());
    using const_reference_t = reference_t;

  public:
    using mixin::get;

    using value_type = cursor::value_type_t<C>;
    using difference_type = cursor::difference_type_t<C>;
    using reference = cursor::reference_t<C>;

    basic_iterator() = default;

    using mixin::mixin;

    constexpr explicit basic_iterator(C&& c)
      noexcept(std::is_nothrow_constructible_v<mixin, C&&>) :
      mixin(std::move(c)) {}


    constexpr explicit basic_iterator(C const& c)
      noexcept(std::is_nothrow_constructible_v<mixin, C const&>) :
      mixin(c) {}

    template <std::convertible_to<C> O>
    constexpr basic_iterator(basic_iterator<O>&& that)
      noexcept(std::is_nothrow_constructible<mixin, O&&>::value) :
      mixin(that.cursor()) {}

    template <std::convertible_to<C> O>
    constexpr basic_iterator(const basic_iterator<O>& that)
      noexcept(std::is_nothrow_constructible<mixin, const O&>::value) :
      mixin(std::move(that.cursor())) {}

    template <std::convertible_to<C> O>
    constexpr basic_iterator& operator=(basic_iterator<O>&& that) &
      noexcept(std::is_nothrow_assignable<C&, O&&>::value) {
      cursor() = std::move(that.cursor());
      return *this;
    }

    template <std::convertible_to<C> O>
    constexpr basic_iterator& operator=(const basic_iterator<O>& that) &
      noexcept(std::is_nothrow_assignable<C&, const O&>::value) {
      cursor() = that.cursor();
      return *this;
    }

    template <class T>
    requires
      (!std::same_as<std::decay_t<T>, basic_iterator> &&
        !cursor::next<C>&&
        cursor::writable<C, T>)
      constexpr basic_iterator& operator=(T&& t) &
      noexcept(noexcept(std::declval<C&>().write(static_cast<T&&>(t)))) {
      cursor() = std::forward<T>(t);
      return *this;
    }

    friend constexpr decltype(auto) iter_move(const basic_iterator& i)
#if !defined(__NVCOMPILER)
      noexcept(noexcept(i.cursor().indirect_move()))
#endif
      requires cursor::indirect_move<C> {
      return i.cursor().indirect_move();
    }

    template <class O>
    requires cursor::indirect_swap<C, O>
      friend constexpr void iter_swap(
        const basic_iterator& x, const basic_iterator<O>& y)
#if !defined(__NVCOMPILER)
      noexcept(noexcept((void)x.indirect_swap(y)))
#endif
    {
      x.indirect_swap(y);
    }

    //Input iterator
    constexpr decltype(auto) operator*() const
      noexcept(noexcept(std::declval<const C&>().read()))
      requires (cursor::readable<C> && !cursor::detail::is_writable_cursor<C>::value) {
      return cursor().read();
    }

    //Output iterator
    constexpr decltype(auto) operator*()
      noexcept(noexcept(reference_t{ cursor() }))
      requires (cursor::next<C>&& cursor::detail::is_writable_cursor<C>::value) {
      return reference_t{ cursor() };
    }

    //Output iterator
    constexpr decltype(auto) operator*() const
      noexcept(noexcept(
        const_reference_t{ cursor() }))
      requires (cursor::next<C>&& cursor::detail::is_writable_cursor<C>::value) {
      return const_reference_t{ cursor() };
    }

    constexpr basic_iterator& operator*() noexcept
      requires (!cursor::next<C>) {
      return *this;
    }

    // operator->: "Manual" deduction override,
    constexpr decltype(auto) operator->() const
      noexcept(noexcept(cursor().arrow()))
      requires cursor::arrow<C> {
      return cursor().arrow();
    }
    // operator->: Otherwise, if reference_t is an lvalue reference,
    constexpr decltype(auto) operator->() const
      noexcept(noexcept(*std::declval<const basic_iterator&>()))
      requires (cursor::readable<C> && !cursor::arrow<C>)
      && std::is_lvalue_reference<const_reference_t>::value{
      return std::addressof(**this);
    }

    // modifiers
    constexpr basic_iterator& operator++() & noexcept {
      return *this;
    }
    constexpr basic_iterator& operator++() &
      noexcept(noexcept(std::declval<basic_iterator>().cursor().next()))
      requires cursor::next<C> {
      cursor().next();
      return *this;
    }

    //C++17 required that *it++ was valid.
    //For input iterators, we can't copy *this, so we need to create a proxy reference.
    constexpr auto operator++(int) &
      noexcept(noexcept(++std::declval<basic_iterator&>()) &&
        std::is_nothrow_move_constructible_v<value_type>&&
        std::is_nothrow_constructible_v<value_type, reference>)
      requires (cursor::single_pass<C>&&
        std::move_constructible<value_type>&&
        std::constructible_from<value_type, reference>) {
      detail::post_increment_proxy<value_type> p(**this);
      ++* this;
      return p;
    }

    //If we can't create a proxy reference, it++ is going to return void
    constexpr void operator++(int) &
      noexcept(noexcept(++std::declval<basic_iterator&>()))
      requires (cursor::single_pass<C> && !(std::move_constructible<value_type>&&
        std::constructible_from<value_type, reference>)) {
      (void)(++(*this));
    }

    //If C is a forward cursor then copying it is fine
    constexpr basic_iterator operator++(int) &
      noexcept(std::is_nothrow_copy_constructible_v<C>&&
        std::is_nothrow_move_constructible_v<C> &&
        noexcept(++std::declval<basic_iterator&>()))
      requires (!cursor::single_pass<C>) {
      auto temp = *this;
      ++* this;
      return temp;
    }

    constexpr basic_iterator& operator--() &
      noexcept(noexcept(cursor().prev()))
      requires cursor::bidirectional<C> {
      cursor().prev();
      return *this;
    }

    //Postfix decrement doesn't have the same issue as postfix increment
    //because bidirectional requires the cursor to be a forward cursor anyway
    //so copying it is fine.
    constexpr basic_iterator operator--(int) &
      noexcept(std::is_nothrow_copy_constructible<basic_iterator>::value&&
        std::is_nothrow_move_constructible<basic_iterator>::value &&
        noexcept(--std::declval<basic_iterator&>()))
      requires cursor::bidirectional<C> {
      auto tmp = *this;
      --* this;
      return tmp;
    }

    constexpr basic_iterator& operator+=(difference_type n) &
      noexcept(noexcept(cursor().advance(n)))
      requires cursor::random_access<C> {
      cursor().advance(n);
      return *this;
    }

    constexpr basic_iterator& operator-=(difference_type n) &
      noexcept(noexcept(cursor().advance(-n)))
      requires cursor::random_access<C> {
      cursor().advance(-n);
      return *this;
    }

    constexpr decltype(auto) operator[](difference_type n) const
      noexcept(noexcept(*(std::declval<basic_iterator&>() + n)))
      requires cursor::random_access<C> {
      return *(*this + n);
    }

    // non-template type-symmetric ops to enable implicit conversions
    friend constexpr difference_type operator-(
      const basic_iterator& x, const basic_iterator& y)
      noexcept(noexcept(y.cursor().distance_to(x.cursor())))
      requires cursor::sized_sentinel_for<C, C> {
      return y.cursor().distance_to(x.cursor());
    }
    friend constexpr bool operator==(
      const basic_iterator& x, const basic_iterator& y)
#if !defined(__NVCOMPILER)
      noexcept(noexcept(x.cursor().equal(y.cursor())))
      requires cursor::sentinel_for<C, C>
#endif
    {
      return x.cursor().equal(y.cursor());
    }
    friend constexpr bool operator!=(
      const basic_iterator& x, const basic_iterator& y)
#if !defined(__NVCOMPILER)
      noexcept(noexcept(!(x == y)))
      requires cursor::sentinel_for<C, C>
#endif
    {
      return !(x == y);
    }
    friend constexpr bool operator<(
      const basic_iterator& x, const basic_iterator& y)
#if !defined(__NVCOMPILER)
      noexcept(noexcept(y - x))
#endif
      requires cursor::sized_sentinel_for<C, C> {
      return 0 < (y - x);
    }
    friend constexpr bool operator>(
      const basic_iterator& x, const basic_iterator& y)
#if !defined(__NVCOMPILER)
      noexcept(noexcept(y - x))
#endif
      requires cursor::sized_sentinel_for<C, C> {
      return 0 > (y - x);
    }
    friend constexpr bool operator<=(
      const basic_iterator& x, const basic_iterator& y)
#if !defined(__NVCOMPILER)
      noexcept(noexcept(y - x))
#endif
      requires cursor::sized_sentinel_for<C, C> {
      return 0 <= (y - x);
    }
    friend constexpr bool operator>=(
      const basic_iterator& x, const basic_iterator& y)
#if !defined(__NVCOMPILER)
      noexcept(noexcept(y - x))
#endif
      requires cursor::sized_sentinel_for<C, C> {
      return 0 >= (y - x);
    }
  };

  namespace detail {
    template <class C>
    struct is_basic_iterator {
      template <class T>
      static auto deduce(basic_iterator<T> const&)->std::true_type;
      template <class T>
      static auto deduce(...)->std::false_type;
      static constexpr inline bool value = decltype(deduce(std::declval<C>()))::value;
    };
  }

  // basic_iterator nonmember functions
  template <class C>
  constexpr basic_iterator<C> operator+(
    const basic_iterator<C>& i, cursor::difference_type_t<C> n)
    noexcept(std::is_nothrow_copy_constructible<basic_iterator<C>>::value&&
      std::is_nothrow_move_constructible<basic_iterator<C>>::value &&
      noexcept(std::declval<basic_iterator<C>&>() += n))
    requires cursor::random_access<C> {
    auto tmp = i;
    tmp += n;
    return tmp;
  }
  template <class C>
  constexpr basic_iterator<C> operator+(
    cursor::difference_type_t<C> n, const basic_iterator<C>& i)
    noexcept(noexcept(i + n))
    requires cursor::random_access<C> {
    return i + n;
  }

  template <class C>
  constexpr basic_iterator<C> operator-(
    const basic_iterator<C>& i, cursor::difference_type_t<C> n)
    noexcept(noexcept(i + (-n)))
    requires cursor::random_access<C> {
    return i + (-n);
  }
  template <class C1, class C2>
  requires cursor::sized_sentinel_for<C1, C2>
    constexpr cursor::difference_type_t<C2> operator-(
      const basic_iterator<C1>& lhs, const basic_iterator<C2>& rhs)
    noexcept(noexcept(
      rhs.get().distance_to(lhs.get()))) {
    return rhs.get().distance_to(lhs.get());
  }
  template <class C, class S>
  requires cursor::sized_sentinel_for<S, C>
    constexpr cursor::difference_type_t<C> operator-(
      const S& lhs, const basic_iterator<C>& rhs)
    noexcept(noexcept(rhs.get().distance_to(lhs))) {
    return rhs.get().distance_to(lhs);
  }
  template <class C, class S>
  requires cursor::sized_sentinel_for<S, C>
    constexpr cursor::difference_type_t<C> operator-(
      const basic_iterator<C>& lhs, const S& rhs)
    noexcept(noexcept(-(rhs - lhs))) {
    return -(rhs - lhs);
  }

  template <class C1, class C2>
  requires cursor::sentinel_for<C2, C1>
    constexpr bool operator==(
      const basic_iterator<C1>& lhs, const basic_iterator<C2>& rhs)
    noexcept(noexcept(lhs.get().equal(rhs.get()))) {
    return lhs.get().equal(rhs.get());
  }
  template <class C, class S>
  requires cursor::sentinel_for<S, C>
    constexpr bool operator==(
      const basic_iterator<C>& lhs, const S& rhs)
    noexcept(noexcept(lhs.get().equal(rhs))) {
    return lhs.get().equal(rhs);
  }
  template <class C, class S>
  requires cursor::sentinel_for<S, C>
    constexpr bool operator==(
      const S& lhs, const basic_iterator<C>& rhs)
    noexcept(noexcept(rhs == lhs)) {
    return rhs == lhs;
  }

  template <class C1, class C2>
  requires cursor::sentinel_for<C2, C1>
    constexpr bool operator!=(
      const basic_iterator<C1>& lhs, const basic_iterator<C2>& rhs)
    noexcept(noexcept(!(lhs == rhs))) {
    return !(lhs == rhs);
  }
  template <class C, class S>
  requires cursor::sentinel_for<S, C>
    constexpr bool operator!=(
      const basic_iterator<C>& lhs, const S& rhs)
    noexcept(noexcept(!lhs.get().equal(rhs))) {
    return !lhs.get().equal(rhs);
  }
  template <class C, class S>
  requires cursor::sentinel_for<S, C>
    constexpr bool operator!=(
      const S& lhs, const basic_iterator<C>& rhs)
    noexcept(noexcept(!rhs.get().equal(lhs))) {
    return !rhs.get().equal(lhs);
  }

  template <class C1, class C2>
  requires cursor::sized_sentinel_for<C1, C2>
    constexpr bool operator<(
      const basic_iterator<C1>& lhs, const basic_iterator<C2>& rhs)
    noexcept(noexcept(lhs - rhs < 0)) {
    return (lhs - rhs) < 0;
  }

  template <class C1, class C2>
  requires cursor::sized_sentinel_for<C1, C2>
    constexpr bool operator>(
      const basic_iterator<C1>& lhs, const basic_iterator<C2>& rhs)
    noexcept(noexcept((lhs - rhs) > 0)) {
    return (lhs - rhs) > 0;
  }

  template <class C1, class C2>
  requires cursor::sized_sentinel_for<C1, C2>
    constexpr bool operator<=(
      const basic_iterator<C1>& lhs, const basic_iterator<C2>& rhs)
    noexcept(noexcept((lhs - rhs) <= 0)) {
    return (lhs - rhs) <= 0;
  }

  template <class C1, class C2>
  requires cursor::sized_sentinel_for<C1, C2>
    constexpr bool operator>=(
      const basic_iterator<C1>& lhs, const basic_iterator<C2>& rhs)
    noexcept(noexcept((lhs - rhs) >= 0)) {
    return (lhs - rhs) >= 0;
  }

  template <class V, bool Const>
  class basic_sentinel {
    using Base = std::conditional_t<Const, const V, V>;

  public:
    std::ranges::sentinel_t<Base> end_{};
    basic_sentinel() = default;
    constexpr explicit basic_sentinel(std::ranges::sentinel_t<Base> end)
      : end_{ std::move(end) } {}

    constexpr basic_sentinel(basic_sentinel<V, !Const> other) requires Const&& std::
      convertible_to<std::ranges::sentinel_t<V>,
      std::ranges::sentinel_t<Base>>
      : end_{ std::move(other.end_) } {}

    constexpr auto end() const {
      return end_;
    }

    friend class basic_sentinel<V, !Const>;
  };

  //tl::compose composes f and g such that compose(f,g)(args...) is f(g(args...)), i.e. g is called first
  template <class F, class G>
  struct compose_fn {
    [[no_unique_address]] F f;
    [[no_unique_address]] G g;

    template <class A, class B>
    compose_fn(A&& a, B&& b) : f(std::forward<A>(a)), g(std::forward<B>(b)) {}

    template <class A, class B, class ... Args>
    static constexpr auto call(A&& a, B&& b, Args&&... args) {
      if constexpr (std::is_void_v<std::invoke_result_t<G, Args...>>) {
        std::invoke(std::forward<B>(b), std::forward<Args>(args)...);
        return std::invoke(std::forward<A>(a));
      }
      else {
        return std::invoke(std::forward<A>(a), std::invoke(std::forward<B>(b), std::forward<Args>(args)...));
      }
    }

    template <class... Args>
    constexpr auto operator()(Args&&... args) & {
      return call(f, g, std::forward<Args>(args)...);
    }

    template <class... Args>
    constexpr auto operator()(Args&&... args) const& {
      return call(f, g, std::forward<Args>(args)...);
    }

    template <class... Args>
    constexpr auto operator()(Args&&... args)&& {
      return call(std::move(f), std::move(g), std::forward<Args>(args)...);
    }

    template <class... Args>
    constexpr auto operator()(Args&&... args) const&& {
      return call(std::move(f), std::move(g), std::forward<Args>(args)...);
    }
  };

  template <class F, class G>
  constexpr auto compose(F&& f, G&& g) {
    return compose_fn<std::remove_cvref_t<F>, std::remove_cvref_t<G>>(std::forward<F>(f), std::forward<G>(g));
  }

  //tl::pipeable takes some invocable and enables:
  //- Piping a single argument to it such that a | pipeable is the same as pipeable(a)
  //- Piping it to another pipeable object, such that a | b is the same as tl::compose(b, a)
  struct pipeable_base {};
  template <class T>
  concept is_pipeable = std::is_base_of_v<pipeable_base, std::remove_cvref_t<T>>;

  template <class F>
  struct pipeable_fn : pipeable_base {
    [[no_unique_address]] F f_;

    constexpr pipeable_fn(F f) : f_(std::move(f)) {}

    template <class... Args>
    constexpr auto operator()(Args&&... args) const requires std::invocable<F, Args...> {
      return std::invoke(f_, std::forward<Args>(args)...);
    }
  };

  template <class F>
  constexpr auto pipeable(F f) {
    return pipeable_fn{ std::move(f) };
  }

  template <class V, class Pipe>
  constexpr auto operator|(V&& v, Pipe&& fn)
    requires (!is_pipeable<V> && is_pipeable<Pipe> && std::invocable<Pipe, V>) {
    return std::invoke(std::forward<Pipe>(fn).f_, std::forward<V>(v));
  }

  template <class Pipe1, class Pipe2>
  constexpr auto operator|(Pipe1&& p1, Pipe2&& p2)
    requires (is_pipeable<Pipe1>&& is_pipeable<Pipe2>) {
    return pipeable(compose(std::forward<Pipe2>(p2).f_, std::forward<Pipe1>(p1).f_));
  }

  //tl::bind_back binds the last N arguments of f to the given ones, returning a new closure
  template <class F, class... Args>
  constexpr auto bind_back(F&& f, Args&&... args) {
    return[f_ = std::forward<F>(f), ...args_ = std::forward<Args>(args)]
    (auto&&... other_args)
    requires std::invocable<F&, decltype(other_args)..., Args&...> {
      return std::invoke(f_, std::forward<decltype(other_args)>(other_args)..., args_...);
    };
  }
}

namespace std {
  template <class C>
  struct iterator_traits<tl::basic_iterator<C>> : tl::cursor::associated_types<C> {};
}

namespace tl  {

  template <class Tuple>
  constexpr inline std::size_t tuple_size = std::tuple_size_v<std::remove_cvref_t<Tuple>>;

  template <std::size_t N>
  using index_constant = std::integral_constant<std::size_t, N>;

  namespace meta {
    //Partially-apply the given template with the given arguments
    template <template <class...> class T, class... Args>
    struct partial {
      template <class... MoreArgs>
      struct apply {
        using type = T<Args..., MoreArgs...>;
      };
    };

    namespace detail {
      template <class T, template<class...> class Into, std::size_t... Idx>
      constexpr auto repeat_into_impl(std::index_sequence<Idx...>)
        ->Into < typename decltype((Idx, std::type_identity<T>{}))::type... > ;
    }

    //Repeat the given type T into the template Into N times
    template <class T, std::size_t N, template <class...> class Into>
    using repeat_into = decltype(tl::meta::detail::repeat_into_impl<T,Into>(std::make_index_sequence<N>{}));
  }

  //If the size of Ts is 2, returns pair<Ts...>, otherwise returns tuple<Ts...>
  namespace detail {
    template<class... Ts>
    struct tuple_or_pair_impl : std::type_identity<std::tuple<Ts...>> {};
    template<class Fst, class Snd>
    struct tuple_or_pair_impl<Fst, Snd> : std::type_identity<std::pair<Fst, Snd>> {};
  }
  template<class... Ts>
  using tuple_or_pair = typename detail::tuple_or_pair_impl<Ts...>::type;

  template <class Tuple>
  constexpr auto min_tuple(Tuple&& tuple) {
    return std::apply([](auto... sizes) {
      return std::ranges::min({
        std::common_type_t<decltype(sizes)...>(sizes)...
        });
      }, std::forward<Tuple>(tuple));
  }

  template <class Tuple>
  constexpr auto max_tuple(Tuple&& tuple) {
    return std::apply([](auto... sizes) {
      return std::ranges::max({
        std::common_type_t<decltype(sizes)...>(sizes)...
        });
      }, std::forward<Tuple>(tuple));
  }

  //Call f on every element of the tuple, returning a new one
  template<class F, class... Tuples>
  constexpr auto tuple_transform(F&& f, Tuples&&... tuples)
  {
    if constexpr (sizeof...(Tuples) > 1) {
      auto call_at_index = []<std::size_t Idx, class Fu, class... Ts>
        (tl::index_constant<Idx>, Fu f, Ts&&... tuples) {
        return f(std::get<Idx>(std::forward<Ts>(tuples))...);
      };

      constexpr auto min_size = tl::min_tuple(std::tuple(tl::tuple_size<Tuples>...));

      return[&] <std::size_t... Idx>(std::index_sequence<Idx...>) {
        return tuple_or_pair < std::decay_t<decltype(call_at_index(tl::index_constant<Idx>{}, std::move(f), std::forward<Tuples>(tuples)...)) > ... >
          (call_at_index(tl::index_constant<Idx>{}, std::move(f), std::forward<Tuples>(tuples)...)...);
      }(std::make_index_sequence<min_size>{});
    }
    else if constexpr (sizeof...(Tuples) == 1) {
      return std::apply([&]<class... Ts>(Ts&&... elements) {
        return tuple_or_pair<std::invoke_result_t<F&, Ts>...>(
          std::invoke(f, std::forward<Ts>(elements))...
          );
      }, std::forward<Tuples>(tuples)...);
    }
    else {
      return std::tuple{};
    }
  }

  //Call f on every element of the tuple
  template<class F, class Tuple>
  constexpr auto tuple_for_each(F&& f, Tuple&& tuple)
  {
    return std::apply([&]<class... Ts>(Ts&&... elements) {
      (std::invoke(f, std::forward<Ts>(elements)), ...);
    }, std::forward<Tuple>(tuple));
  }

  template <class Tuple>
  constexpr auto tuple_pop_front(Tuple&& tuple) {
    return std::apply([](auto&& head, auto&&... tail) {
      return std::pair(std::forward<decltype(head)>(head), std::tuple(std::forward<decltype(tail)>(tail)...));
      }, std::forward<Tuple>(tuple));
  }

  namespace detail {
    template <class F, class V>
    constexpr auto tuple_fold_impl(F, V v) {
      return v;
    }
    template <class F, class V, class Arg, class... Args>
    constexpr auto tuple_fold_impl(F f, V v, Arg arg, Args&&... args) {
      return tl::detail::tuple_fold_impl(f,
        std::invoke(f, std::move(v), std::move(arg)),
        std::forward<Args>(args)...);
    }
  }

  template <class F, class T, class Tuple>
  constexpr auto tuple_fold(Tuple tuple, T t, F f) {
    return std::apply([&](auto&&... args) {
      return tl::detail::tuple_fold_impl(std::move(f), std::move(t), std::forward<decltype(args)>(args)...);
      }, std::forward<Tuple>(tuple));
  }

  template<class... Tuples>
  constexpr auto tuple_zip(Tuples&&... tuples) {
    auto zip_at_index = []<std::size_t Idx, class... Ts>
      (tl::index_constant<Idx>, Ts&&... tuples) {
      return tuple_or_pair<std::decay_t<decltype(std::get<Idx>(std::forward<Ts>(tuples)))>...>(std::get<Idx>(std::forward<Ts>(tuples))...);
    };

    constexpr auto min_size = tl::min_tuple(std::tuple(tl::tuple_size<Tuples>...));

    return[&] <std::size_t... Idx>(std::index_sequence<Idx...>) {
      return tuple_or_pair<std::decay_t<decltype(zip_at_index(tl::index_constant<Idx>{}, std::forward<Tuples>(tuples)...))>... >
        (zip_at_index(tl::index_constant<Idx>{}, std::forward<Tuples>(tuples)...)...);
    }(std::make_index_sequence<min_size>{});
  }

  template <std::ranges::forward_range... Vs>
  requires (std::ranges::view<Vs> && ...) class cartesian_product_view
    : public std::ranges::view_interface<cartesian_product_view<Vs...>> {

    //random access + sized is allowed because we can jump all iterators to the end
    template <class... Ts>
    static constexpr bool am_common = (std::ranges::common_range<Ts> && ...)
      || ((std::ranges::random_access_range<Ts> && ...) && (std::ranges::sized_range<Ts> && ...));

    template <class... Ts>
    static constexpr bool am_sized = (std::ranges::sized_range<Ts> && ...);

    //requires common because we need to be able to cycle the iterators from begin to end in O(n)
    template <class... Ts>
    static constexpr bool am_bidirectional =
      ((std::ranges::bidirectional_range<Ts> && ...) && (std::ranges::common_range<Ts> && ...));

    //requires sized because we need to calculate new positions for iterators using arithmetic modulo the range size
    template <class... Ts>
    static constexpr bool am_random_access =
      ((std::ranges::random_access_range<Ts> && ...) &&
        (std::ranges::sized_range<Ts> && ...));

    template <class... Ts>
    static constexpr bool am_distanceable =
      ((std::sized_sentinel_for<std::ranges::iterator_t<Ts>, std::ranges::iterator_t<Ts>>) && ...)
      && am_sized<Ts...>;

    std::tuple<Vs...> bases_;

    template <bool Const>
    class cursor;

    //Wraps the end iterator for the 0th range.
    //This is all that's required because the cursor will only ever set the 0th iterator to end when
    //the cartesian product operation has completed.
    template <bool Const>
    class sentinel {
      using parent = std::conditional_t<Const, const cartesian_product_view, cartesian_product_view>;
      using first_base = decltype(std::get<0>(std::declval<parent>().bases_));
      std::ranges::sentinel_t<first_base> end_;

    public:
      sentinel() = default;
      sentinel(std::ranges::sentinel_t<first_base> end) : end_(std::move(end)) {}

      //const-converting constructor
      constexpr sentinel(sentinel<!Const> other) requires Const &&
        (std::convertible_to<std::ranges::sentinel_t<first_base>, std::ranges::sentinel_t<const first_base>>)
        : end_(std::move(other.end_)) {
      }

      template <bool>
      friend class cursor;
    };

    template <bool Const>
    class cursor {
      template<class T>
      using constify = std::conditional_t<Const, const T, T>;

      template<class T>
      using intify = std::conditional_t<true, std::int64_t, T>;
      // Instead of storing a pointer to the views, we'll store the sentinels:
      // constify<std::tuple<Vs...>>* bases_;
      std::tuple<std::ranges::iterator_t<constify<Vs>>...> currents_{};
      std::tuple<std::ranges::iterator_t<constify<Vs>>...> begins_{};
      std::tuple<std::ranges::sentinel_t<constify<Vs>>...> ends_{};
      std::tuple<intify<Vs>...> counts_{};
      std::int64_t idx_;

    public:
      using reference =
        std::tuple<std::ranges::range_reference_t<constify<Vs>>...>;
      using value_type =
        std::tuple<std::ranges::range_value_t<constify<Vs>>...>;

      using difference_type = std::ptrdiff_t;

      cursor() = default;
      constexpr explicit cursor(constify<std::tuple<Vs...>>* bases)
        : currents_( tl::tuple_transform(std::ranges::begin, *bases) )
        , begins_( currents_ )
        , ends_( tl::tuple_transform(std::ranges::end, *bases) )
        , counts_( tl::tuple_transform(std::ranges::size, *bases) )
        , idx_(0)
      {}

      //If the underlying ranges are common, we can get to the end by assigning from end
      constexpr explicit cursor(as_sentinel_t, constify<std::tuple<Vs...>>* bases)
        requires(std::ranges::common_range<Vs> && ...)
        : cursor{ bases }
      {
        std::get<0>(currents_) = std::get<0>(ends_);
      }

      //If the underlying ranges are sized and random access, we can get to the end by moving it forward by size
      constexpr explicit cursor(as_sentinel_t, constify<std::tuple<Vs...>>* bases)
        requires(!(std::ranges::common_range<Vs> && ...) && (std::ranges::random_access_range<Vs> && ...) && (std::ranges::sized_range<Vs> && ...))
        : cursor{ bases }
      {
        std::get<0>(currents_) += std::ranges::size(std::ranges::subrange(std::get<0>(begins_), std::get<0>(ends_)));
      }

      //const-converting constructor
      constexpr cursor(cursor<!Const> i) requires Const && (std::convertible_to<
        std::ranges::iterator_t<Vs>,
        std::ranges::iterator_t<constify<Vs>>> && ...)
        : currents_{ std::move(i.currents_) } {}


      constexpr decltype(auto) read() const {
        return tuple_transform([](auto& i) -> decltype(auto) { return *i; }, currents_);
      }

      template <std::size_t N = (sizeof...(Vs) - 1)>
      void update(int idx) {
        if constexpr(N == 0)
          std::get<N>(currents_) = idx + std::get<N>(begins_);
        else
          std::get<N>(currents_) = idx % std::get<N>(counts_) + std::get<N>(begins_);
        if constexpr (N > 0) {
          idx /= std::get<N>(counts_);
          update<N-1>(idx);
        }
      }

      //Increment the iterator at std::get<N>(currents_)
      //If that iterator hits its end, recurse to std::get<N-1>
      void next() {
        advance(1);
      }

      //Decrement the iterator at std::get<N>(currents_)
      //If that iterator was at its begin, cycle it to end and recurse to std::get<N-1>
      void prev() requires (am_bidirectional<constify<Vs>...>) {
        advance(-1);
      }

      void advance(difference_type n) requires (am_random_access<constify<Vs>...>) {
        idx_ += n;
        update(idx_);
      }

      constexpr bool equal(const cursor& rhs) const
#if !defined(__NVCOMPILER)
        requires (std::equality_comparable<std::ranges::iterator_t<constify<Vs>>> && ...)
#endif
      {
        return currents_ == rhs.currents_;
      }

      constexpr bool equal(const sentinel<Const>& s) const {
        return std::get<0>(currents_) == s.end_;
      }

      template <std::size_t N = (sizeof...(Vs) - 1)>
      constexpr auto distance_to(cursor const& other) const
        requires (am_distanceable<constify<Vs>...>) {
        if constexpr (N == 0) {
          return std::ranges::distance(std::get<0>(currents_), std::get<0>(other.currents_));
        }
        else {
          auto distance = distance_to<N - 1>(other);
          auto scale = std::ranges::distance(std::get<N>(begins_), std::get<N>(ends_));
          auto diff = std::ranges::distance(std::get<N>(currents_), std::get<N>(other.currents_));
#if !defined(__NVCOMPILER)
          return difference_type{ distance * scale + diff };
#else
          return static_cast<difference_type>(distance * scale + diff);
#endif
        }
      }

      friend class cursor<!Const>;
    };

  public:
    cartesian_product_view() = default;
    explicit cartesian_product_view(Vs... bases) : bases_(std::move(bases)...) {}

    constexpr auto begin() requires (!(simple_view<Vs> && ...)) {
      return basic_iterator{ cursor<false>(std::addressof(bases_)) };
    }
    constexpr auto begin() const requires (std::ranges::range<const Vs> && ...) {
      return basic_iterator{ cursor<true>(std::addressof(bases_)) };
    }

    constexpr auto end() requires (!(simple_view<Vs> && ...) && am_common<Vs...>) {
      return basic_iterator{ cursor<false>(as_sentinel, std::addressof(bases_)) };
    }

    constexpr auto end() const requires (am_common<const Vs...>) {
      return basic_iterator{ cursor<true>(as_sentinel, std::addressof(bases_)) };
    }

    constexpr auto end() requires(!(simple_view<Vs> && ...) && !am_common<Vs...>) {
      return sentinel<false>(std::ranges::end(std::get<0>(bases_)));
    }

    constexpr auto end() const requires ((std::ranges::range<const Vs> && ...) && !am_common<const Vs...>) {
      return sentinel<true>(std::ranges::end(std::get<0>(bases_)));
    }

    constexpr auto size() requires (am_sized<Vs...>) {
      //Multiply all the sizes together, returning the common type of all of them
      return std::apply([](auto&&... bases) {
        using size_type = std::common_type_t<std::ranges::range_size_t<decltype(bases)>...>;
        return (static_cast<size_type>(std::ranges::size(bases)) * ...);
        }, bases_);
    }

    constexpr auto size() const requires (am_sized<const Vs...>) {
      return std::apply([](auto&&... bases) {
        using size_type = std::common_type_t<std::ranges::range_size_t<decltype(bases)>...>;
        return (static_cast<size_type>(std::ranges::size(bases)) * ...);
        }, bases_);
    }
  };

#ifndef DISABLE_CART_PROD_IOTA_SPEC
  template <typename W, typename B>
  requires std::is_integral_v<W> && std::is_integral_v<B>
  class cartesian_product_view<
    std::ranges::iota_view<W, B>,
    std::ranges::iota_view<W, B>,
    std::ranges::iota_view<W, B>
  >
    : public std::ranges::view_interface<
      cartesian_product_view<
        std::ranges::iota_view<W, B>,
        std::ranges::iota_view<W, B>,
        std::ranges::iota_view<W, B>
      >
    > {

    using self = cartesian_product_view<
      std::ranges::iota_view<W, B>,
      std::ranges::iota_view<W, B>,
      std::ranges::iota_view<W, B>
    >;
  public:
    W x_b, y_b, z_b;
    B nx, ny, nz;
  private:

    template <bool Const>
    class cursor;

    template <bool Const>
    class cursor {
      W x_i, y_i, z_i, x_0, y_0, z_0;
      B nx, ny, nz;

    public:
      using reference = std::tuple<W const&, W const&, W const&>;
      using value_type = std::tuple<W, W, W>;

      using difference_type = std::ptrdiff_t;

      cursor() = default;
      constexpr explicit cursor(
        W x,
        W y,
        W z,
        B nx,
        B ny,
        B nz)
      :
        x_i(x),
        y_i(y),
        z_i(z),
        x_0(x),
        y_0(y),
        z_0(z),
        nx(nx),
        ny(ny),
        nz(nz)
      {}

      constexpr decltype(auto) read() const {
        return std::make_tuple<W const&, W const&, W const&>(x_i, y_i, z_i);
      }
      constexpr auto linear() const {
        return (z_i - z_0) + nz * (y_i - y_0) + nz * ny * (x_i - x_0);
      }

      void advance(difference_type n) {
        auto idx = linear() + n;
        x_i = idx / (nz * ny) + x_0;
        y_i = (idx / nz) % ny + y_0;
        z_i = idx % nz + z_0;
      }
      void next() {
        //advance(1);
        ++z_i;
        if (z_i == (z_0 + nz)) {
          z_i = z_0;
          ++y_i;
          if (y_i == (y_0 + ny)) {
            y_i = y_0;
            ++x_i;
          }
        }
      }
      void prev(){
        advance(-1);
      }

      constexpr bool equal(const cursor& rhs) const {
        return x_i == rhs.x_i && y_i == rhs.y_i && z_i == rhs.z_i;
      }

      constexpr auto distance_to(cursor const& other) const {
        auto idx = linear();
        auto oidx = other.linear();
        return static_cast<difference_type>(oidx) - static_cast<difference_type>(idx);
      }

      friend class cursor<!Const>;
    };

  public:
    cartesian_product_view() = default;
    constexpr explicit cartesian_product_view(
      std::ranges::iota_view<W, B> xs,
      std::ranges::iota_view<W, B> ys,
      std::ranges::iota_view<W, B> zs
    ) : x_b(*std::ranges::begin(xs))
      , y_b(*std::ranges::begin(ys))
      , z_b(*std::ranges::begin(zs))
      , nx(std::ranges::size(xs))
      , ny(std::ranges::size(ys))
      , nz(std::ranges::size(zs))
    {}

    constexpr auto begin() {
      return basic_iterator{ cursor<false>(x_b, y_b, z_b, nx, ny, nz) };
    }
    constexpr auto begin() const {
      return basic_iterator{ cursor<true>(x_b, y_b, z_b, nx, ny, nz) };
    }

    constexpr auto size() {
      return nx * ny * nz;
    }

    constexpr auto size() const {
      return nx * ny * nz;
    }

    constexpr auto end() {
      return begin() + size();
    }

    constexpr auto end() const {
      return begin() + size();
    }
  };

  template <typename W, typename B>
  requires std::is_integral_v<W> && std::is_integral_v<B>
  class cartesian_product_view<
    std::ranges::iota_view<W, B>,
    std::ranges::iota_view<W, B>
  >
    : public std::ranges::view_interface<
      cartesian_product_view<
        std::ranges::iota_view<W, B>,
        std::ranges::iota_view<W, B>
      >
    > {

    using self = cartesian_product_view<
      std::ranges::iota_view<W, B>,
      std::ranges::iota_view<W, B>
    >;
  public:
    W x_b, y_b;
    B nx, ny;
  private:

    template <bool Const>
    class cursor;

    template <bool Const>
    class cursor {
      W x_i, y_i, x_0, y_0;
      B nx, ny;

    public:
      using reference = std::tuple<W const&, W const&>;
      using value_type = std::tuple<W, W>;

      using difference_type = std::ptrdiff_t;

      cursor() = default;
      constexpr explicit cursor(
        W x,
        W y,
        B nx,
        B ny)
      :
        x_i(x),
        y_i(y),
        x_0(x),
        y_0(y),
        nx(nx),
        ny(ny)
      {}

      constexpr decltype(auto) read() const {
        return std::make_tuple<W const&, W const&>(x_i, y_i);
      }
      constexpr auto linear() const {
        return (y_i - y_0) + ny * (x_i - x_0);
      }

      void advance(difference_type n) {
        auto idx = linear() + n;
        x_i = idx / ny + x_0;
        y_i = idx % ny + y_0;
      }
      void next() {
        //advance(1);
        ++y_i;
        if (y_i == (y_0 + ny)) {
          y_i = y_0;
          ++x_i;
        }
      }
      void prev(){
        advance(-1);
      }

      constexpr bool equal(const cursor& rhs) const {
        return x_i == rhs.x_i && y_i == rhs.y_i;
      }

      constexpr auto distance_to(cursor const& other) const {
        auto idx = linear();
        auto oidx = other.linear();
        return static_cast<difference_type>(oidx) - static_cast<difference_type>(idx);
      }

      friend class cursor<!Const>;
    };

  public:
    cartesian_product_view() = default;
    constexpr explicit cartesian_product_view(
      std::ranges::iota_view<W, B> xs,
      std::ranges::iota_view<W, B> ys
    ) : x_b(*std::ranges::begin(xs))
      , y_b(*std::ranges::begin(ys))
      , nx(std::ranges::size(xs))
      , ny(std::ranges::size(ys))
    {}

    constexpr auto begin() {
      return basic_iterator{ cursor<false>(x_b, y_b, nx, ny) };
    }
    constexpr auto begin() const {
      return basic_iterator{ cursor<true>(x_b, y_b, nx, ny) };
    }

    constexpr auto size() {
      return nx * ny;
    }

    constexpr auto size() const {
      return nx * ny;
    }

    constexpr auto end() {
      return begin() + size();
    }

    constexpr auto end() const {
      return begin() + size();
    }

  };

  template <typename W, typename B>
  requires std::is_integral_v<W> && std::is_integral_v<B>
  class cartesian_product_view<
    std::ranges::iota_view<W, B>
  >
    : public std::ranges::view_interface<
      cartesian_product_view<
        std::ranges::iota_view<W, B>
      >
    > {

    int x_i, x_e;

    template <bool Const>
    class cursor;

    template <bool Const>
    class cursor {
      long x_i, x_e;

    public:
      using reference = std::tuple<int const&>;
      using value_type = std::tuple<int>;

      using difference_type = std::ptrdiff_t;

      cursor() = default;
      constexpr explicit cursor(
        int x, int x_e)
      :
        x_i(x),
        x_e(x_e)
      {}

      constexpr decltype(auto) read() const {
        return std::make_tuple<int const&>(x_i);
      }
      void advance(difference_type n) {
        x_i += n;
      }
      void next() {
        advance(1);
      }
      void prev(){
        advance(-1);
      }

      constexpr bool equal(const cursor& rhs) const {
        return x_i == rhs.x_i;
      }

      constexpr auto distance_to(cursor const& other) const {
        return static_cast<difference_type>(other.x_i - x_i);
      }

      friend class cursor<!Const>;
    };

  public:
    cartesian_product_view() = default;
    constexpr explicit cartesian_product_view(
      std::ranges::iota_view<W, B> xs
    ) : x_i(*std::ranges::begin(xs))
      , x_e(*std::ranges::begin(xs) + std::ranges::size(xs))
    {}

    constexpr auto begin() {
      return basic_iterator{ cursor<false>(x_i, x_e) };
    }
    constexpr auto begin() const {
      return basic_iterator{ cursor<true>(x_i, x_e) };
    }

    constexpr auto size() {
      return x_e - x_i;
    }

    constexpr auto size() const {
      return x_e - x_i;
    }

    constexpr auto end() {
      return begin() + size();
    }

    constexpr auto end() const {
      return begin() + size();
    }
  };
#endif // DISABLE_CART_PROD_IOTA_SPEC

  template <class... Rs>
  cartesian_product_view(Rs&&...)->cartesian_product_view<std::views::all_t<Rs>...>;

  namespace views {
    namespace detail {
      class cartesian_product_fn {
      public:
        constexpr std::ranges::empty_view<std::tuple<>> operator()() const noexcept {
          return {};
        }
#ifndef DISABLE_CART_PROD_IOTA_SPEC
        template <typename W, typename B>
        constexpr auto operator()(
          std::ranges::iota_view<W, B> xs
        ) const {
          return tl::cartesian_product_view<
            std::ranges::iota_view<W, B>
          >{ std::move(xs) };
        }

        template <typename W, typename B>
        constexpr auto operator()(
          std::ranges::iota_view<W, B> xs,
          std::ranges::iota_view<W, B> ys
        ) const {
          return tl::cartesian_product_view<
            std::ranges::iota_view<W, B>,
            std::ranges::iota_view<W, B>
          >{ std::move(xs), std::move(ys) };
        }
        template <typename W, typename B>
        constexpr auto operator()(
          std::ranges::iota_view<W, B> xs,
          std::ranges::iota_view<W, B> ys,
          std::ranges::iota_view<W, B> zs
        ) const {
          return tl::cartesian_product_view<
            std::ranges::iota_view<W, B>,
            std::ranges::iota_view<W, B>,
            std::ranges::iota_view<W, B>
          >{ std::move(xs), std::move(ys), std::move(zs) };
        }

#endif
        template <std::ranges::viewable_range... V>
        requires ((std::ranges::forward_range<V> && ...) && (sizeof...(V) != 0))
          constexpr auto operator()(V&&... vs) const {
          return tl::cartesian_product_view{ std::views::all(std::forward<V>(vs))... };
        }
      };
    }  // namespace detail

    inline constexpr detail::cartesian_product_fn cartesian_product;
  }  // namespace views

}  // namespace tl

////////////////////////////////////////////////////////////////////////////////
// Strided View

namespace tl {

  template <std::ranges::forward_range V>
  requires std::ranges::view<V> class stride_view
    : public std::ranges::view_interface<stride_view<V>> {
  private:
    //Cannot be common for non-sized bidirectional ranges because calculating the predecessor to end would be O(n).
    template <class T>
    static constexpr bool am_common = std::ranges::common_range<T> &&
      (!std::ranges::bidirectional_range<T> || std::ranges::sized_range<T>);

    template <class T> static constexpr bool am_sized = std::ranges::sized_range<T>;

    V base_;
    std::ranges::range_difference_t<V> stride_size_;

    //The cursor for stride_view may need to keep track of additional state.
    //Consider the case where you have a vector of 0,1,2 and you stride with a size of 2.
    //If you take the end iterator for the stride_view and decrement it, then you should end up at 2, not at 1.
    //As such, there's additional work required to track where to decrement to in the case that the underlying range is bidirectional.
    //This is handled by a bunch of base classes for the cursor.

    //Case when underlying range is not bidirectional, i.e. we don't care about calculating offsets
    template <bool Const, class Base = std::conditional_t<Const, const V, V>, bool = std::ranges::bidirectional_range<Base>, bool = std::ranges::sized_range<Base>>
    struct cursor_base {
      cursor_base() = default;
      constexpr explicit cursor_base(std::ranges::range_difference_t<Base> offs) {}

      //These are both no-ops
      void set_offset(std::ranges::range_difference_t<Base> off) {}
      std::ranges::range_difference_t<Base> get_offset() {
        return 0;
      }
    };

    //Case when underlying range is bidirectional but not sized. We need to keep track of the offset if we hit the end iterator.
    template <bool Const, class Base>
    struct cursor_base<Const, Base, true, false> {
      using difference_type = std::ranges::range_difference_t<Base>;
      difference_type offset_{};

      cursor_base() = default;
      constexpr explicit cursor_base(difference_type offset)
        : offset_{ offset } {}

      void set_offset(difference_type off) {
        offset_ = off;
      }

      difference_type get_offset() {
        return offset_;
      }
    };

    //Case where underlying is bidirectional and sized. We can calculate offsets from the end on construction.
    template <bool Const, class Base>
    struct cursor_base<Const, Base, true, true> {
      using difference_type = std::ranges::range_difference_t<Base>;
      difference_type offset_{};

      cursor_base() = default;
      constexpr explicit cursor_base(difference_type offset)
        : offset_{ offset } {}

      //No-op because we're precomputing the offset
      void set_offset(std::ranges::range_difference_t<Base>) {}

      std::ranges::range_difference_t<Base> get_offset() {
        return offset_;
      }
    };

    template <bool Const>
    struct cursor : cursor_base<Const> {
      template <class T>
      using constify = std::conditional_t<Const, const T, T>;
      using Base = constify<V>;

      using difference_type = std::ranges::range_difference_t<Base>;

      std::ranges::iterator_t<Base> current_{};
      std::ranges::sentinel_t<Base> end_{};
      std::ranges::range_difference_t<Base> stride_size_{};

      cursor() = default;

      //Pre-calculate the offset for sized ranges
      constexpr cursor(std::ranges::iterator_t<Base> begin, Base* base, std::ranges::range_difference_t<Base> stride_size)
        requires(std::ranges::sized_range<Base>)
        : cursor_base<Const>(stride_size - (std::ranges::size(*base) % stride_size)),
        current_(std::move(begin)), end_(std::ranges::end(*base)), stride_size_(stride_size) {}

      constexpr cursor(std::ranges::iterator_t<Base> begin, Base* base, std::ranges::range_difference_t<Base> stride_size)
        requires(!std::ranges::sized_range<Base>)
        : cursor_base<Const>(), current_(std::move(begin)), end_(std::ranges::end(*base)), stride_size_(stride_size) {}

      //const-converting constructor
      constexpr cursor(cursor<!Const> i) requires Const&& std::convertible_to<
        std::ranges::iterator_t<V>,
        std::ranges::iterator_t<const V>>
        : cursor_base<Const>(i.get_offset()) {}

      constexpr decltype(auto) read() const {
        return *current_;
      }

      constexpr void next() {
        auto delta = std::ranges::advance(current_, stride_size_, end_);
        //This will track the amount we actually moved by last advance,
        //which will be less than the stride size if range_size % stride_size != 0
        this->set_offset(delta);
      }

      constexpr void prev() requires std::ranges::bidirectional_range<Base> {
        auto delta = -stride_size_;
        //If we're at the end we may need to offset the amount to move back by
        if (current_ == end_) {
          delta += this->get_offset();
        }
        std::advance(current_, delta);
      }

      constexpr void advance(difference_type x)
        requires std::ranges::random_access_range<Base> {
        if (x == 0) return;

        x *= stride_size_;

        if (x > 0) {
          auto delta = std::ranges::advance(current_, x, end_); // TODO: submit PR with this bugfix
          this->set_offset(delta);
        }
        else if (x < 0) {
          if (current_ == end_) {
            x += this->get_offset();
          }
          std::advance(this->current_, x);  // TODO: submit PR with this bugfix
        }
      }

      // TODO: submit PR with distance_to
      constexpr auto distance_to(cursor const& other) const
        // am_distanceable<V>:
        requires (std::sized_sentinel_for<std::ranges::iterator_t<V>, std::ranges::iterator_t<V>>)
          && std::ranges::sized_range<V>
      {
        auto delta = std::ranges::distance(this->current_, other.current_);
        if (delta < 0)
          delta -= stride_size_ - 1;
        else
          delta += stride_size_ -1;
        return delta / stride_size_;
      }

      constexpr bool equal(cursor const& rhs) const {
        return current_ == rhs.current_;
      }
      constexpr bool equal(basic_sentinel<V, Const> const& rhs) const {
        return current_ == rhs.end_;
      }

      friend struct cursor<!Const>;
    };

  public:
    stride_view() = default;
    stride_view(V v, std::ranges::range_difference_t<V> n) : base_(std::move(v)), stride_size_(n) {}

    constexpr auto begin() requires (!simple_view<V>) {
      return basic_iterator{ cursor<false>{ std::ranges::begin(base_), std::addressof(base_), stride_size_ } };
    }

    constexpr auto begin() const requires (std::ranges::range<const V>) {
      return basic_iterator{ cursor<true>{ std::ranges::begin(base_), std::addressof(base_), stride_size_ } };
    }

    constexpr auto end() requires (!simple_view<V>&& am_common<V>) {
      return basic_iterator{ cursor<false>(std::ranges::end(base_), std::addressof(base_), stride_size_) };
    }

    constexpr auto end() const requires (std::ranges::range<const V>&& am_common<const V>) {
      return basic_iterator{ cursor<true>(std::ranges::end(base_), std::addressof(base_), stride_size_) };
    }

    constexpr auto end() requires (!simple_view<V> && !am_common<V>) {
      return basic_sentinel<V, false>(std::ranges::end(base_));
    }

    constexpr auto end() const requires (std::ranges::range<const V> && !am_common<const V>) {
      return basic_sentinel<V, true>(std::ranges::end(base_));
    }

    constexpr auto size() requires (am_sized<V>) {
      return (std::ranges::size(base_) + stride_size_ - 1) / stride_size_;
    }

    constexpr auto size() const requires (am_sized<const V>) {
      return (std::ranges::size(base_) + stride_size_ - 1) / stride_size_;
    }

    auto& base() {
      return base_;
    }

    auto const& base() const {
      return base_;
    }
  };

  template <class R, class N>
  stride_view(R&&, N n)->stride_view<std::views::all_t<R>>;

  namespace views {
    namespace detail {
      struct stride_fn_base {
        template <std::ranges::viewable_range R>
        constexpr auto operator()(R&& r, std::ranges::range_difference_t<R> n) const
          requires std::ranges::forward_range<R> {
          return stride_view(std::forward<R>(r), n);
        }
      };

      struct stride_fn : stride_fn_base {
        using stride_fn_base::operator();

        template <std::integral N>
        constexpr auto operator()(N n) const {
          return pipeable(bind_back(stride_fn_base{}, n));
        }
      };
    }

    constexpr inline detail::stride_fn stride;
  }
}

namespace std::ranges {
  template <class R>
  inline constexpr bool enable_borrowed_range<tl::stride_view<R>> = enable_borrowed_range<R>;
}

////////////////////////////////////////////////////////////////////////////////
// std::ranges::views re-export:

namespace std {
  namespace ranges {
    using tl::cartesian_product_view;
    namespace views {
      inline constexpr tl::views::detail::cartesian_product_fn cartesian_product;
      inline constexpr tl::views::detail::stride_fn stride;
    } // namespace views
  } // namespace ranges
} // namespace std
