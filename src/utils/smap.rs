use std::collections::BTreeMap;

pub trait SMapAccum<T: Clone> {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, T) -> Result<(A, T), E>
    ) -> Result<(A, Self), E> where Self: Sized;

    fn smap_accum_l<A>(self, acc: A, f: impl Fn(A, T) -> (A, T)) -> (A, Self) where Self: Sized {
        self.smap_accum_l_result(Ok(acc), |acc, t| Ok::<(A, T), ()>(f(acc, t)))
            .unwrap()
    }

    fn smap_result<E>(
        self,
        f: impl Fn(T) -> Result<T, E>
    ) -> Result<Self, E> where Self: Sized {
        let (_, t) = self.smap_accum_l_result(Ok(()), |_, t| Ok(((), f(t)?)))?;
        Ok(t)
    }

    fn smap(self, f: impl Fn(T) -> T) -> Self where Self: Sized {
        let (_, res) = self.smap_accum_l::<()>((), |_, x| ((), f(x)));
        res
    }
}

pub trait SFold<T: Clone> {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &T) -> Result<A, E>
    ) -> Result<A, E> where Self: Sized;

    fn sfold_owned_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, T) -> Result<A, E>
    ) -> Result<A, E> where Self: Sized {
        self.sfold_result(acc, |acc, t| f(acc, t.clone()))
    }

    fn sfold_owned<A>(self, acc: A, f: impl Fn(A, T) -> A) -> A where Self: Sized {
        self.sfold_result(Ok(acc), |acc, t| Ok::<A, ()>(f(acc, t.clone()))).unwrap()
    }

    fn sfold<A>(&self, acc: A, f: impl Fn(A, &T) -> A) -> A where Self: Sized {
        self.sfold_result(Ok(acc), |acc, t| Ok::<A, ()>(f(acc, t))).unwrap()
    }
}

impl<T: Clone> SMapAccum<T> for Vec<T> {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, T) -> Result<(A, T), E>
    ) -> Result<(A, Self), E> {
        self.into_iter()
            .fold(Ok((acc?, vec![])), |acc, x| {
                let (acc, mut elems) = acc?;
                let (acc, x) = f(acc, x)?;
                elems.push(x);
                Ok((acc, elems))
            })
    }
}

impl<T: Clone, K> SMapAccum<T> for Vec<(K, T)> {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, T) -> Result<(A, T), E>,
    ) -> Result<(A, Self), E> {
        self.into_iter()
            .fold(Ok((acc?, vec![])), |acc, (k, v)| {
                let (acc, mut elems) = acc?;
                let (acc, v) = f(acc, v)?;
                elems.push((k, v));
                Ok((acc, elems))
            })
    }
}

impl<T: Clone, I: Ord> SMapAccum<T> for BTreeMap<I, T> {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, T) -> Result<(A, T), E>
    ) -> Result<(A, Self), E> {
        self.into_iter()
            .fold(Ok((acc?, BTreeMap::new())), |acc, (k, t)| {
                let (acc, mut elems) = acc?;
                let (acc, t) = f(acc, t)?;
                elems.insert(k, t);
                Ok((acc, elems))
            })
    }
}

impl<T: Clone> SMapAccum<T> for Option<Box<T>> {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, T) -> Result<(A, T), E>
    ) -> Result<(A, Self), E> {
        match self {
            Some(e) => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Some(Box::new(e))))
            },
            None => Ok((acc?, None))
        }
    }
}

impl<T: Clone> SFold<T> for Vec<T> {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &T) -> Result<A, E>
    ) -> Result<A, E> {
        self.iter().fold(acc, |acc, t| f(acc?, t))
    }
}

impl<T: Clone, K> SFold<T> for Vec<(K, T)> {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &T) -> Result<A, E>
    ) -> Result<A, E> {
        self.iter().fold(acc, |acc, (_, t)| f(acc?, t))
    }
}

impl<T: Clone, I> SFold<T> for BTreeMap<I, T> {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &T) -> Result<A, E>
    ) -> Result<A, E> {
        self.iter()
            .fold(acc, |acc, (_, t)| f(acc?, t))
    }
}

impl<T: Clone> SFold<T> for Option<Box<T>> {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &T) -> Result<A, E>
    ) -> Result<A, E> {
        match self {
            Some(e) => f(acc?, e),
            None => acc
        }
    }
}
