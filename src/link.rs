//! Light linking: which lights light which meshes. Hydra hands each light
//! the token of its lightLink collection, empty when it lights everything,
//! and each mesh the tokens of the collections it is in. Each distinct
//! collection gets one bit of a mask: a light carries its collection's bit
//! (`light::Light::link`), each instance its mesh's bits
//! (`transform::Instance`), and the light loop skips a light whose bit the
//! mesh lacks (`shaders/frame.slang`).

use std::collections::BTreeMap;
use std::sync::Once;

use crate::store::{Light, Mesh, Store};

/// The collections a mask holds bits for.
const BITS: usize = 32;

/// Each linked light's collection by its bit.
pub struct Links {
    bits: BTreeMap<String, u32>,
}

impl Links {
    /// Bits in token order, so they do not hang on the order lights were
    /// added in. Past `BITS` collections, a light lights every mesh.
    pub fn new(store: &Store) -> Self {
        let mut tokens: Vec<&str> = (store.lights().values())
            .filter_map(|l| l.link.as_deref())
            .filter(|l| !l.is_empty())
            .collect();
        tokens.sort_unstable();
        tokens.dedup();
        if tokens.len() > BITS {
            static WARNED: Once = Once::new();
            WARNED.call_once(|| {
                eprintln!(
                    "pika: warning: the lights link {} collections, and Pika links {BITS}; the lights of the rest light every mesh (warned once)",
                    tokens.len()
                )
            });
        }
        let bits = (tokens.into_iter().take(BITS).enumerate())
            .map(|(i, t)| (t.to_owned(), 1 << i))
            .collect();
        Self { bits }
    }

    /// The light's bit, or 0 when it lights every mesh.
    pub fn light(&self, light: &Light) -> u32 {
        let link = light.link.as_deref().unwrap_or("");
        self.bits.get(link).copied().unwrap_or(0)
    }

    /// The bits of the collections the mesh is in.
    pub fn mesh(&self, mesh: &Mesh) -> u32 {
        (mesh.categories.iter().flatten())
            .filter_map(|c| self.bits.get(c))
            .fold(0, |mask, bit| mask | bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether a light of mask `light` lights a mesh of mask `mesh`, as the
    /// light loop decides it.
    fn reaches(light: u32, mesh: u32) -> bool {
        light & mesh == light
    }

    fn store(links: &[&str], categories: &[&[&str]]) -> (Store, Vec<u64>, Vec<u64>) {
        let mut store = Store::new();
        let lights = (links.iter().enumerate())
            .map(|(i, link)| {
                let id = store.insert_light(format!("/light{i}"));
                store.set_light_link(id, (*link).into()).unwrap();
                id
            })
            .collect();
        let meshes = (categories.iter().enumerate())
            .map(|(i, c)| {
                let id = store.insert_mesh(format!("/mesh{i}"));
                let c = c.iter().map(|&t| t.to_owned()).collect();
                store.set_categories(id, c).unwrap();
                id
            })
            .collect();
        (store, lights, meshes)
    }

    /// Which lights light which meshes, row by mesh.
    fn lit(store: &Store, lights: &[u64], meshes: &[u64]) -> Vec<Vec<bool>> {
        let links = Links::new(store);
        (meshes.iter())
            .map(|m| {
                let mesh = links.mesh(&store.meshes()[m]);
                (lights.iter())
                    .map(|l| reaches(links.light(&store.lights()[l]), mesh))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn lights_light_the_meshes_of_their_collection() {
        // Two lights share one collection; one lights everything.
        let (store, lights, meshes) = store(
            &["group_1", "", "group_0", "group_1"],
            &[&["group_0"], &["group_1"], &[], &["group_0", "group_1"]],
        );
        let t = true;
        let f = false;
        assert_eq!(
            lit(&store, &lights, &meshes),
            [[f, t, t, f], [t, t, f, t], [f, t, f, f], [t, t, t, t]]
        );
    }

    #[test]
    fn categories_no_light_links_light_nothing_more() {
        let (store, lights, meshes) = store(&["group_0", ""], &[&["group_9"], &[]]);
        assert_eq!(
            lit(&store, &lights, &meshes),
            [[false, true], [false, true]]
        );
        // A mesh not yet synced is in no collection.
        let mut store = store;
        let unsynced = store.insert_mesh("/unsynced".into());
        assert_eq!(Links::new(&store).mesh(&store.meshes()[&unsynced]), 0);
    }

    #[test]
    fn bits_follow_tokens_not_light_order() {
        let (a, la, _) = store(&["b", "a"], &[]);
        let (b, lb, _) = store(&["a", "b"], &[]);
        let bit = |s: &Store, id: u64| Links::new(s).light(&s.lights()[&id]);
        assert_eq!(bit(&a, la[0]), bit(&b, lb[1]));
        assert_ne!(bit(&a, la[0]), bit(&a, la[1]));
    }

    #[test]
    fn lights_past_the_last_bit_light_every_mesh() {
        let tokens: Vec<String> = (0..=BITS).map(|i| format!("group_{i:02}")).collect();
        let links: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let (store, lights, meshes) = store(&links, &[&[]]);
        let lit = &lit(&store, &lights, &meshes)[0];
        assert_eq!(lit.iter().filter(|&&l| l).count(), 1);
        assert!(lit[BITS]);
    }

    #[test]
    fn links_move_the_draws() {
        let (mut store, lights, meshes) = store(&["group_0"], &[&[]]);
        store.take_moved();
        store.set_light_link(lights[0], "group_0".into()).unwrap();
        store.set_categories(meshes[0], Vec::new()).unwrap();
        assert!(!store.take_moved().draws, "an unchanged link moves nothing");
        store.set_light_link(lights[0], "group_1".into()).unwrap();
        assert!(store.take_moved().draws);
        store
            .set_categories(meshes[0], vec!["group_1".into()])
            .unwrap();
        assert!(store.take_moved().draws);
        store.remove_light(lights[0]).unwrap();
        assert!(store.take_moved().draws);
    }
}
